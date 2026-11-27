"""Quick check of the refusal system without a GPU (fake models where a real one is needed).

    python check_refusal.py

Checks the four methods of refusal.py.
Run calibrate_refusal.py --method bm25 first (it creates data/refusal_bm25.json).
"""

import types

import numpy as np
import pandas as pd

import refusal as R
from retrieval import load_bm25, load_embedding_matrix

passed = 0


def ok(name, condition, detail=""):
    global passed
    print(f"{'OK  ' if condition else 'FAIL'} {name} {detail}")
    passed += bool(condition)


# ---------- 1. BM25 (real index, real calibrated model) ----------
print("\n== 1. BM25 ==")
bm25 = load_bm25()
refuser = R.ScoreRefuser.load("bm25")
for question in ["What are the requirements for a blocked account in Germany?",
                 "How much money is currently available in my blocked account?",
                 "What is the best pizza in Naples?"]:
    d = refuser.decide(R.bm25_features(question, bm25))
    print(f"     refuse={str(d.refuse):5s} P={d.p_unanswerable:.2f} (cut-off {d.threshold:.2f})  {question}")
ok("bm25 features", set(R.bm25_features("test", bm25)) == set(R.BM25_FEATURES))

# ---------- 2. Dense (real chunk vectors, fake query encoder) ----------
print("\n== 2. Dense ==")
_, matrix = load_embedding_matrix()


class FakeEncoder:
    """Returns the vector of chunk 0, so the best match must be ~1.0."""
    def encode(self, texts, **kwargs):
        return matrix[:1]


f = R.dense_features("test", matrix, FakeEncoder())
ok("dense features", set(f) == set(R.DENSE_FEATURES), f"top1={f['top1']:.3f} (expected ~1.0)")

# ---------- 3. LLM judge (fake tokenizer and model) ----------
print("\n== 3. LLM judge ==")
try:
    import torch

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [{"YES": 1, "NO": 2}[text]]

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, texts, return_tensors="pt"):
            class Batch(dict):
                def to(self, device):
                    return self
            return Batch(input_ids=torch.zeros(1, 4, dtype=torch.long))

    class FakeModel:
        device = "cpu"

        def __call__(self, **kwargs):
            logits = torch.zeros(1, 4, 5)
            logits[0, -1, 1] = 2.0          # "YES" more likely than "NO"
            return types.SimpleNamespace(logits=logits)

    judge = R.LLMJudge(FakeTokenizer(), FakeModel())
    p_yes = judge.p_answerable("q?", pd.DataFrame({"text": ["chunk a", "chunk b"]}))
    ok("llm judge P(YES)", abs(p_yes - 1 / (1 + np.exp(-2))) < 1e-4, f"p_yes={p_yes:.3f} (expected 0.881)")
except ImportError:
    print("SKIP torch not installed here; the LLM judge is checked in Colab")

# ---------- 4. Self-refusal ----------
print("\n== 4. Self-refusal ==")
ok("detects refusal", R.is_self_refusal("I don't have enough information to answer that."))
ok("ignores normal answer", not R.is_self_refusal("The fee is 58 euros per semester [1]."))

# ---------- Refusal in the pipeline: refusal happens before generation ----------
print("\n== Refusal pipeline ==")
import pipeline as P
from data import fetch_chunk_citations

calls = []                                       # every real retrieval is recorded here
P.RAGPipeline.retrieve = lambda self, q: calls.append(q) or fetch_chunk_citations([1, 2])
P.generate = lambda **kwargs: ("", "Students need a residence permit to study in Germany [1].")

rag = P.RAGPipeline.__new__(P.RAGPipeline)       # no models loaded
rag.verbose, rag.tokenizer, rag.model = False, None, None
rag.citation_retries, rag.strict_citations = 0, True

rag.refusal = types.SimpleNamespace(check=lambda q, ctx: R.RefusalDecision(True, "bm25", 0.9, 0.5))
_, result = rag.answer_question("anything")
ok("refuses before generation", result.refused and result.refusal_reason == "bm25", f"-> {result.answer}")

rag.refusal = types.SimpleNamespace(check=lambda q, ctx: R.RefusalDecision(False, "bm25", 0.1, 0.5))
calls.clear()
_, result = rag.answer_question("anything")
ok("answers when not refused", not result.refused, f"-> {result.answer[:60]}")
ok("retrieval runs only once", len(calls) == 1, f"(calls={len(calls)})")

rag.refusal = None                               # refusal_method="none": prompt refusal only
_, result = rag.answer_question("anything")
ok("no refusal check when method is none", not result.refused and rag.last_refusal is None)

print(f"\n{passed} checks passed")
