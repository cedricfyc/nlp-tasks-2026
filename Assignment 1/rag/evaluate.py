"""LLM-as-judge evaluation of RAG answers against the ground truth.

The judge (prompt, JSON parsing, retries, resumable SQLite storage) is the working
code from Test Scripts/llm_judge_for_rag_evaluation.ipynb, moved into a module so
run_eval.py can use it. Scores per dimension are 0-5; an answer "succeeds" when the
mean score * 20 >= threshold (default 60).

    python evaluate.py --answers ../results/rag_both_refusal-none.db
    python evaluate.py --answers ../results/rag_both_refusal-none.db --summary-only

Normally run through run_eval.py (which generates the answers and then judges them).
"""

import argparse
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

METRICS = ("correctness", "coverage", "consistency", "groundedness", "relevance")
JUDGE_MAX_NEW_TOKENS = 4096   # the judge only emits a short JSON object
JUDGE_MAX_ATTEMPTS = 3        # retries per row if the JSON is invalid
JUDGE_TEMPERATURE = 0.1       # low temperature -> more stable scoring
SUCCESS_THRESHOLD = 60        # mean score * 20 must reach this to count as a success

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")

JUDGE_SYSTEM_PROMPT = '''
You are a strict but fair evaluator comparing a candidate answer against a
ground-truth reference. Evaluate meaning, not wording: accept paraphrases,
equivalent explanations, and concise answers.

Score each dimension with an integer from 0 to 5:

- correctness: Information is factually consistent with the ground truth.
- coverage: Essential points, conditions, and recommendations are included.
- consistency: No contradictions of the ground truth (5 = none, 0 = pervasive).
- groundedness: No unsupported, speculative, or fabricated claims (5 = none).
- relevance: The answer addresses the actual question.

Scale: 5 = fully satisfied; 4 = minor gaps; 3 = partially satisfied;
2 = major gaps; 1 = barely satisfied; 0 = not satisfied.

Rules:
- Weigh omissions and errors by importance.
- Use "primary_domain" first; for multi-domain questions, also assess
  coverage of the other listed "domains" as emphasized in the ground truth.
- If "is_unanswerable" is true, a clear acknowledgement of insufficient
  information is correct; penalize fabricated or speculative personal
  information under correctness and groundedness.
- Do not reward keyword matching without semantic correctness.
- Provide "reasons" as a list of concise strings explaining the scores.
  Include one string for each dimension, identifying important errors,
  omissions, contradictions, unsupported claims, or strengths as applicable.

Return ONLY valid JSON (double quotes, no trailing commas, no extra text):

{
  "correctness": 0,
  "coverage": 0,
  "consistency": 0,
  "groundedness": 0,
  "relevance": 0,
  "reasons": [
    "Correctness: ...",
    "Coverage: ...",
    "Consistency: ...",
    "Groundedness: ...",
    "Relevance: ..."
  ]
}

EXAMPLES

QUESTION:
"I am an international student who has started a paid internship.
My insurer says my contributions may change. Why can employment
affect my insurance?"

GROUND TRUTH:
"Employment type, hours, remuneration and whether an internship
is mandatory or voluntary can affect social-insurance treatment.
Student status alone does not settle every case. Give the insurer
and employer accurate study and employment details and request
a written classification before estimating take-home pay."

Example 1 — Fully correct answer

CANDIDATE ANSWER:
"Your insurance contributions can depend on the type of employment,
working hours, and salary. Mandatory and voluntary internships may
also be treated differently. Being a student does not automatically
determine your insurance status, so you should provide your insurer
and employer with accurate details and request written confirmation
of your classification."

YOUR EXPECTED OUTPUT:
{
  "correctness": 5,
  "coverage": 5,
  "consistency": 5,
  "groundedness": 5,
  "relevance": 5,
  "reasons": [
    "Correctness: The explanation aligns with the reference.",
    "Coverage: It includes the key factors and recommended actions.",
    "Consistency: It does not contradict the reference.",
    "Groundedness: It makes no unsupported claims.",
    "Relevance: It directly answers the question."
  ]
}

Example 2 — Partially correct answer

CANDIDATE ANSWER:
"Your insurance contributions may change because you are earning
money through employment. You should contact your insurer to
check whether your contributions will increase."

YOUR EXPECTED OUTPUT:
{
  "correctness": 3,
  "coverage": 1,
  "consistency": 5,
  "groundedness": 5,
  "relevance": 3,
  "reasons": [
    "Correctness: It gives a plausible but incomplete explanation.",
    "Coverage: It omits most classification factors and recommended steps.",
    "Consistency: It does not contradict the reference.",
    "Groundedness: Its claims are supported by the reference.",
    "Relevance: It addresses the question but gives limited explanation."
  ]
}

Example 3 — Incorrect answer

CANDIDATE ANSWER:
"Your insurance contributions cannot change because you are a
student. All international students are automatically exempt
from social-insurance contributions, regardless of their salary,
working hours, or type of internship."

YOUR EXPECTED OUTPUT:
{
  "correctness": 0,
  "coverage": 0,
  "consistency": 0,
  "groundedness": 0,
  "relevance": 1,
  "reasons": [
    "Correctness: The claimed automatic exemption is false.",
    "Coverage: It omits the essential explanation and recommendations.",
    "Consistency: It directly contradicts the reference.",
    "Groundedness: Its blanket exemption claim is unsupported.",
    "Relevance: It addresses the topic but fails to answer correctly."
  ]
}
'''.strip()

def log(message):
    """Timestamped, flushed terminal output."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# =====================================================================
# Prompt and parsing
# =====================================================================
def make_compare_prompt(user_input, ground_truth, candidate_answer, metadata):
    """Build a comparison prompt for one candidate answer."""
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
    return f"""
Metadata:
{metadata_json}

USER INPUT:
{user_input}

GROUND TRUTH:
{ground_truth}

CANDIDATE LLM ANSWER:
{candidate_answer}

Compare the candidate with the ground truth using the rubric.
Return only the required JSON.
""".strip()


def parse_judge_json(raw):
    """Validate judge output. Returns (scores dict, reasons as JSON text)."""
    if not isinstance(raw, str):
        raise TypeError("Judge response must be a string.")

    raw = _SPECIAL_TOKEN_RE.sub("", raw)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)

    start, end = raw.find("{"), raw.rfind("}")      # tolerate stray text around the JSON
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in judge response.")
    try:
        scores_json = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(scores_json, dict):
        raise ValueError("Judge response must be a JSON object.")

    scores = {}
    for metric in METRICS:
        score = scores_json.get(metric)
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
            raise ValueError(f"{metric} must be an integer between 0 and 5.")
        scores[metric] = score

    reasons = scores_json.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
        raise ValueError("reasons must be a list of strings.")
    return scores, json.dumps(reasons, ensure_ascii=False)


# =====================================================================
# Judge
# =====================================================================
class Judge:
    """Scores answers with a judge LLM. Pass an already loaded (tokenizer, model) to reuse it."""

    def __init__(self, tokenizer=None, model=None, system_prompt=JUDGE_SYSTEM_PROMPT):
        if tokenizer is None or model is None:
            from generation import load_model
            tokenizer, model = load_model()
        self.tokenizer, self.model = tokenizer, model
        self.system_prompt = system_prompt

    def judge(self, question, ground_truth, answer, metadata=None):
        """Returns (scores, reasons_json); retries if the JSON is invalid."""
        from generation import generate

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": make_compare_prompt(question, ground_truth, answer,
                                                            metadata or {})},
        ]
        last_error = None
        for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
            _, raw = generate(tokenizer=self.tokenizer, model=self.model, messages=messages,
                              user_temperature=JUDGE_TEMPERATURE, think=False,
                              max_new_tokens=JUDGE_MAX_NEW_TOKENS)
            try:
                return parse_judge_json(raw)
            except (ValueError, TypeError) as exc:
                last_error = exc
                log(f"    invalid judge output (attempt {attempt}/{JUDGE_MAX_ATTEMPTS}): {exc}")
        raise ValueError(f"Judge failed after {JUDGE_MAX_ATTEMPTS} attempts: {last_error}")


# =====================================================================
# Storage (same table as the notebook's judge_evaluation.db, resumable)
# =====================================================================
JUDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_results (
    id                  INTEGER PRIMARY KEY,
    question_id         TEXT,
    rag_algorithm       TEXT NOT NULL,
    top_k               INTEGER NOT NULL CHECK (top_k > 0),
    reranking_method    TEXT NOT NULL,
    selected_chunks     TEXT NOT NULL,
    correctness         INTEGER NOT NULL CHECK (correctness BETWEEN 0 AND 5),
    coverage            INTEGER NOT NULL CHECK (coverage BETWEEN 0 AND 5),
    consistency         INTEGER NOT NULL CHECK (consistency BETWEEN 0 AND 5),
    groundedness        INTEGER NOT NULL CHECK (groundedness BETWEEN 0 AND 5),
    relevance           INTEGER NOT NULL CHECK (relevance BETWEEN 0 AND 5),
    reasons             TEXT NOT NULL,
    success             INTEGER NOT NULL CHECK (success IN (0, 1)),
    created_at          TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_eval_run_question
    ON evaluation_results (question_id, rag_algorithm, top_k, reranking_method);
"""


def load_answers(db_path, table="rag_results") -> pd.DataFrame:
    """Answers written by run_eval.py (or the notebook's rag_evaluation.db)."""
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def judge_answers(answers: pd.DataFrame, judge_db, judge: Judge, rag_algorithm, top_k,
                  reranking_method, threshold=SUCCESS_THRESHOLD) -> int:
    """Judge every answer not yet in judge_db for this configuration. Returns rows stored."""
    Path(judge_db).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(judge_db)) as conn:
        conn.executescript(JUDGE_SCHEMA)
        done = {r[0] for r in conn.execute(
            "SELECT question_id FROM evaluation_results WHERE rag_algorithm=? AND top_k=? "
            "AND reranking_method=?", (rag_algorithm, top_k, reranking_method))}

    todo = answers[~answers["id"].astype(str).isin(done)]
    log(f"Judging {len(todo)} of {len(answers)} answers ({len(answers) - len(todo)} already done)")
    stored, failed, started = 0, [], time.time()

    for position, row in enumerate(todo.itertuples(index=False), start=1):
        row = row._asdict()
        metadata = {k: row.get(k) for k in ("id", "type", "domains", "primary_domain",
                                            "is_unanswerable", "fetched_context")}
        try:
            scores, reasons = judge.judge(row["question"], row["ground_truth"],
                                          row["response"] or "", metadata)
        except KeyboardInterrupt:
            log("Interrupted; finished rows are saved, run again to resume.")
            break
        except Exception as exc:
            failed.append(row["id"])
            log(f"[{position}/{len(todo)}] {row['id']} FAILED: {exc}")
            continue

        success = sum(scores.values()) / len(scores) * 20 >= threshold
        with closing(sqlite3.connect(judge_db)) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO evaluation_results (question_id, rag_algorithm, top_k, "
                "reranking_method, selected_chunks, correctness, coverage, consistency, "
                "groundedness, relevance, reasons, success, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(row["id"]), rag_algorithm, int(top_k), reranking_method,
                 row.get("fetched_context") or "[]", *[scores[m] for m in METRICS], reasons,
                 int(success), datetime.now(timezone.utc).isoformat()))
        stored += 1
        log(f"[{position}/{len(todo)}] {row['id']} " + " ".join(f"{m[:4]}={scores[m]}" for m in METRICS)
            + f" -> {'PASS' if success else 'FAIL'} ({time.time() - started:.0f}s)")

    if failed:
        log(f"Failed ids (run again to retry): {failed}")
    return stored


def load_judged(answers_db, judge_db) -> pd.DataFrame:
    """Answers joined with their judge scores."""
    answers = load_answers(answers_db)
    with closing(sqlite3.connect(judge_db)) as conn:
        scores = pd.read_sql_query("SELECT * FROM evaluation_results", conn)
    scores = scores.drop(columns=["id"]).rename(columns={"question_id": "id"})
    return answers.merge(scores, on="id", how="left")


def summarize(judged: pd.DataFrame) -> dict:
    """Mean scores and success rate, overall and per question type."""
    evaluated = judged.dropna(subset=["correctness"])
    if evaluated.empty:
        print("No judge results found. Run the judge first.")
        return {}
    overall = {m: round(float(evaluated[m].mean()), 2) for m in METRICS}
    overall["success_rate"] = round(float(evaluated["success"].mean()), 3)
    overall["judge_score_100"] = round(float(evaluated[list(METRICS)].mean(axis=1).mean() * 20), 1)
    overall["n"] = int(len(evaluated))
    by_type = (evaluated.groupby("type")[list(METRICS) + ["success"]].mean().round(2)
               .to_dict(orient="index"))

    print(f"\nJudged answers: {overall['n']}")
    print(f"Mean judge score: {overall['judge_score_100']}/100   "
          f"success rate (score >= {SUCCESS_THRESHOLD}): {overall['success_rate']:.0%}")
    print("Means: " + ", ".join(f"{m}={overall[m]}" for m in METRICS))
    print(evaluated.groupby("type")[list(METRICS) + ["success"]].mean().round(2).to_string())
    return {"overall": overall, "by_type": by_type}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, help="rag_<setting>.db written by run_eval.py")
    parser.add_argument("--judge-db", help="default: judge_<setting>.db next to the answers")
    parser.add_argument("--threshold", type=int, default=SUCCESS_THRESHOLD)
    parser.add_argument("--summary-only", action="store_true", help="only print the summary")
    args = parser.parse_args()

    answers_db = Path(args.answers)
    judge_db = Path(args.judge_db or answers_db.with_name(answers_db.name.replace("rag_", "judge_", 1)))
    if not args.summary_only:
        answers = load_answers(answers_db)
        setting = answers["setting"].iloc[0] if "setting" in answers else answers_db.stem
        top_k = int(answers["top_k"].iloc[0]) if "top_k" in answers else 3
        judge_answers(answers, judge_db, Judge(), setting, top_k,
                      answers["reranking_method"].iloc[0] if "reranking_method" in answers else "RRF",
                      args.threshold)
    summarize(load_judged(answers_db, judge_db))
