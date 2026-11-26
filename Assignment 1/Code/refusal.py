"""Refusal system: decides before generation whether a question should be refused.

Four methods: BM25 score, dense score, LLM judge, and the generator's own refusal (system prompt).
The cut-off is learned from labelled questions (calibrate_refusal.py), not a fixed score limit.
"""
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from config import DATA_DIR, TOP_K
from retrieval import bm25_tokenise

# Number of chunks the features look at (more than TOP_K so a "gap" can be measured)
FEATURE_DEPTH = 5


@dataclass
class RefusalDecision:
    refuse: bool
    method: str
    p_unanswerable: float          # 0..1, higher = more likely to refuse
    threshold: float
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# 1. BM25 features
# =====================================================================
BM25_FEATURES = ["norm_top1", "coverage_top1", "coverage_topk", "relative_gap", "log_top1"]


def bm25_features(query: str, bm25_payload, depth: int = FEATURE_DEPTH) -> dict:
    """Query-normalised features from the BM25 scores over the whole corpus."""
    bm25 = bm25_payload["bm25"]
    tokens = bm25_tokenise(query)
    scores = np.asarray(bm25.get_scores(tokens))
    order = np.argsort(scores)[::-1][:depth]
    top = scores[order]

    # IDF of each distinct query word (negative IDF = very common word -> ignore)
    idf = {t: max(float(bm25.idf.get(t, 0.0)), 0.0) for t in set(tokens)}
    idf_sum = sum(idf.values()) or 1e-9
    # Highest BM25 score this query could reach: every word with saturated term frequency
    max_possible = idf_sum * (bm25.k1 + 1)

    def coverage(doc_indices) -> float:
        """IDF-weighted share of query words that appear in these chunks."""
        found = set()
        for i in doc_indices:
            found |= {t for t in idf if t in bm25.doc_freqs[i]}
        return sum(idf[t] for t in found) / idf_sum

    top1 = float(top[0]) if top.size else 0.0
    return {
        "norm_top1": top1 / max_possible,
        "coverage_top1": coverage(order[:1]),
        "coverage_topk": coverage(order[:TOP_K]),
        "relative_gap": (top1 - float(top[-1])) / top1 if top1 > 0 else 0.0,
        "log_top1": float(np.log1p(top1)),
    }


# =====================================================================
# 2. Dense (embedding) features
# =====================================================================
DENSE_FEATURES = ["top1", "mean_topk", "gap_1_5", "z_top1"]


def dense_features(query: str, vector_matrix, embedding_model, depth: int = FEATURE_DEPTH) -> dict:
    """Features from the cosine similarities between the query and every chunk."""
    q = embedding_model.encode([query], prompt_name="query", normalize_embeddings=True,
                               convert_to_numpy=True)[0].astype(np.float32)
    scores = vector_matrix @ q
    top = np.sort(scores)[::-1][:depth]
    return {
        "top1": float(top[0]),
        "mean_topk": float(top[:TOP_K].mean()),
        "gap_1_5": float(top[0] - top[-1]),
        # how far the best chunk stands out from the corpus as a whole
        "z_top1": float((top[0] - scores.mean()) / (scores.std() + 1e-9)),
    }


# =====================================================================
# Calibrated score -> decision (used by bm25, dense and llm_judge)
# =====================================================================
class ScoreRefuser:
    """Logistic regression on the features + a cut-off chosen by cross-validation."""

    def __init__(self, method: str, feature_names: list[str], mean=None, std=None,
                 coef=None, intercept=0.0, threshold=0.5, metrics=None):
        self.method = method
        self.feature_names = feature_names
        self.mean = np.asarray(mean if mean is not None else np.zeros(len(feature_names)))
        self.std = np.asarray(std if std is not None else np.ones(len(feature_names)))
        self.coef = np.asarray(coef if coef is not None else np.zeros(len(feature_names)))
        self.intercept = float(intercept)
        self.threshold = float(threshold)
        self.metrics = metrics or {}

    # ---------- training ----------
    @classmethod
    def fit(cls, method, feature_names, X, y, max_false_refusal=0.10, folds=5, seed=0):
        """X: (n, f) features, y: 1 = unanswerable.

        The cut-off maximises balanced accuracy on out-of-fold probabilities, subject
        to refusing at most `max_false_refusal` of the answerable questions
        (wrongly refusing a good question is worse than answering a bad one,
        which the citation check can still catch).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)

        def model():
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(class_weight="balanced", C=1.0))

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        oof = cross_val_predict(model(), X, y, cv=cv, method="predict_proba")[:, 1]
        threshold, stats = choose_threshold(oof, y, max_false_refusal)
        stats["cv_roc_auc"] = float(roc_auc_score(y, oof))
        stats["n_questions"], stats["n_unanswerable"] = int(len(y)), int(y.sum())

        final = model().fit(X, y)          # refit on all labelled questions
        scaler, lr = final[0], final[1]
        return cls(method, feature_names, scaler.mean_, scaler.scale_, lr.coef_[0],
                   lr.intercept_[0], threshold, stats), oof

    # ---------- use ----------
    def probability(self, features: dict) -> float:
        x = np.array([features[name] for name in self.feature_names], dtype=float)
        z = ((x - self.mean) / self.std) @ self.coef + self.intercept
        return float(1 / (1 + np.exp(-z)))

    def decide(self, features: dict) -> RefusalDecision:
        p = self.probability(features)
        return RefusalDecision(p >= self.threshold, self.method, p, self.threshold, features)

    # ---------- save / load ----------
    def save(self, path=None):
        path = Path(path or DATA_DIR / f"refusal_{self.method}.json")
        path.write_text(json.dumps({
            "method": self.method, "feature_names": self.feature_names,
            "mean": self.mean.tolist(), "std": self.std.tolist(), "coef": self.coef.tolist(),
            "intercept": self.intercept, "threshold": self.threshold, "metrics": self.metrics,
        }, indent=2), encoding="utf-8")
        print(f"Saved {path}")
        return path

    @classmethod
    def load(cls, method, path=None):
        path = Path(path or DATA_DIR / f"refusal_{method}.json")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run calibrate_refusal.py --method {method}.")
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return cls(cfg["method"], cfg["feature_names"], cfg["mean"], cfg["std"], cfg["coef"],
                   cfg["intercept"], cfg["threshold"], cfg.get("metrics"))


def choose_threshold(p, y, max_false_refusal=0.10):
    """Pick the cut-off on P(unanswerable). Returns (threshold, stats at that cut-off)."""
    p, y = np.asarray(p), np.asarray(y)
    best = None
    for t in np.unique(np.concatenate([p, [1.0]])):
        refuse = p >= t
        recall = refuse[y == 1].mean()              # unanswerable questions refused
        false_refusal = refuse[y == 0].mean()       # answerable questions refused
        bal_acc = (recall + (1 - false_refusal)) / 2
        if false_refusal <= max_false_refusal and (best is None or bal_acc > best[1]):
            best = (float(t), bal_acc, recall, false_refusal)
    t, bal_acc, recall, false_refusal = best
    return t, {"threshold": t, "balanced_accuracy": float(bal_acc),
               "refusal_recall": float(recall), "false_refusal_rate": float(false_refusal),
               "max_false_refusal": max_false_refusal}


# =====================================================================
# 3. LLM as judge
# =====================================================================
JUDGE_PROMPT = """You check a retrieval system for international students in Germany.
Decide whether the SOURCES contain the information needed to answer the QUESTION.
Answer NO if the question asks for personal or private data (the user's own account,
application, contract, appointment, medical record, a decision about their individual
case) or anything the sources do not state. Otherwise answer YES.
Reply with one word: YES or NO.

QUESTION:
{question}

SOURCES:
{sources}"""


class LLMJudge:
    """Asks an LLM 'can these chunks answer the question?' and reads P(YES) from the logits.

    Reuses the generator already loaded by the pipeline (no second model needed);
    any chat model from transformers works.
    """

    def __init__(self, tokenizer, model, max_chars_per_chunk=1200):
        self.tokenizer, self.model = tokenizer, model
        self.max_chars = max_chars_per_chunk
        self.yes_id = tokenizer.encode("YES", add_special_tokens=False)[0]
        self.no_id = tokenizer.encode("NO", add_special_tokens=False)[0]

    def p_answerable(self, question: str, context_df) -> float:
        import torch

        sources = "\n\n".join(f"[{i}] {str(text)[:self.max_chars]}"
                              for i, text in enumerate(context_df["text"], start=1))
        messages = [{"role": "user",
                     "content": JUDGE_PROMPT.format(question=question, sources=sources)}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, [self.yes_id, self.no_id]].float()
        return float(torch.softmax(logits, dim=0)[0])

    def features(self, question, context_df) -> dict:
        return {"p_yes": self.p_answerable(question, context_df)}


LLM_JUDGE_FEATURES = ["p_yes"]


# =====================================================================
# 4. Self-refusal (system prompt only)
# =====================================================================
_SELF_REFUSAL = re.compile(r"(?:don't|do not) have enough information", re.IGNORECASE)


def is_self_refusal(answer: str) -> bool:
    """True if the generator refused by itself (generation.SYSTEM_PROMPT rule)."""
    return bool(_SELF_REFUSAL.search(answer or ""))


# =====================================================================
# One entry point for the pipeline
# =====================================================================
class RefusalSystem:
    """Wraps one method; call check() after retrieval and before generation."""

    def __init__(self, method: str, pipeline):
        if method not in {"bm25", "dense", "llm_judge"}:
            raise ValueError("method must be 'bm25', 'dense' or 'llm_judge' "
                             "('self' needs no extra step)")
        self.method, self.pipeline = method, pipeline
        self.refuser = ScoreRefuser.load(method)
        self.judge = LLMJudge(pipeline.tokenizer, pipeline.model) if method == "llm_judge" else None

    def features(self, question, context_df) -> dict:
        p = self.pipeline
        if self.method == "bm25":
            return bm25_features(question, p.bm25_payload)
        if self.method == "dense":
            return dense_features(question, p.vector_matrix, p.embedding_model)
        return self.judge.features(question, context_df)

    def check(self, question, context_df) -> RefusalDecision:
        return self.refuser.decide(self.features(question, context_df))
