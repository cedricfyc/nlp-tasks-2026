"""Ground-truth vs. LLM answer evaluation (work in progress by the team).

Recommended method: an LLM-as-judge rubric for the main success score, with
embedding cosine similarity as a supporting diagnostic. Two correct answers may
use different wording, so cosine similarity alone is not a reliable metric.
The judge returns 0-5 per dimension, converted to 0-100.

TODO (team) - open points carried over from evaluate_rag.ipynb, code unchanged:
  - EvaluationPipeline.judge_candidate_answer and judge_llm are not implemented yet.
  - RAG_OUTPUTS_PATH is not set (answers still need to be generated and saved).
  - combine_data: the loop uses rag_output_data instead of rag_output_entry,
    and test items have no "retrieved_contexts" field.
  - compute_cosine_similarity reads "ground_truth", but combine_data names it "reference".
  - parse_judge_json expects "score", but the prompt returns five dimension scores;
    the JSON example in the prompt has a trailing comma.
  - run_evaluation: SUCCESS_THRESHOLD is undefined and "return" is inside the loop.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd

from config import DATASET_PATH, EMBEDDING_MODEL_NAME
from generation import ask, load_model

# Evaluation data file. Question, ground truth, category, etc.
TESTING_DATA_PATH = DATASET_PATH

# Outputs by the LLM
RAG_OUTPUTS_PATH = None


def load_json_records(path: str) -> json:
    """Load JSON records from a provided path"""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def combine_data(test_data_path: str, rag_outputs_path: str) -> pd.DataFrame:
    test_data = load_json_records(path=test_data_path)
    rag_output_data = load_json_records(path=rag_outputs_path)

    # Create ground-truth map for concatenating to the RAG outputs dict
    response_map = {}

    for rag_output_entry in rag_output_data:
        response_map[rag_output_data["question"]] = rag_output_data["ground_truth"]

    # Combine datasets
    combined_data = []

    # Keep within RAGAS format
    for output in test_data:
        combined_item = {
            "id": output["id"],
            "type": output["type"],
            "domains": output["domains"],
            "primary_domain": output["primary_domain"],
            "user_input": output["question"],
            "retrieved_contexts": output["retrieved_contexts"],
            # Output from the RAG system
            "response": response_map.get(output["ground_truth"], ""),
            "reference": output["ground_truth"],
            "is_unanswerable": output["is_unanswerable"]
        }

        combined_data.append(combined_item)
    combined_df = pd.DataFrame(combined_data)
    combined_df[["id", "type", "domains", "primary_domain", "user_input", "response", "reference", "is_unanswerable"]].head()

    return combined_df


def compute_cosine_similarity(combined_df: pd.DataFrame):
    from sentence_transformers import SentenceTransformer

    EMBEDDING_MODEL = EMBEDDING_MODEL_NAME

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    refs = combined_df["ground_truth"].fillna("").astype(str).tolist()
    preds = combined_df["response"].fillna("").astype(str).tolist()
    ref_vecs = embedder.encode(refs, normalize_embeddings=True, show_progress_bar=True)
    pred_vecs = embedder.encode(preds, normalize_embeddings=True, show_progress_bar=True)
    combined_df["embedding_similarity"] = (ref_vecs * pred_vecs).sum(axis=1)
    combined_df[["id", "embedding_similarity"]].head()


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

Return ONLY valid JSON (double quotes, no trailing commas, no extra text):

{
  "correctness": 0,
  "coverage": 0,
  "consistency": 0,
  "groundedness": 0,
  "relevance": 0,
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
{"correctness": 5, "coverage": 5, "consistency": 5, "groundedness": 5, "relevance": 5}

Example 2 — Partially correct answer

CANDIDATE ANSWER:
"Your insurance contributions may change because you are earning
money through employment. You should contact your insurer to
check whether your contributions will increase."

YOUR EXPECTED OUTPUT:
{"correctness": 3, "coverage": 1, "consistency": 5, "groundedness": 5, "relevance": 3}

Example 3 — Incorrect answer

CANDIDATE ANSWER:
"Your insurance contributions cannot change because you are a
student. All international students are automatically exempt
from social-insurance contributions, regardless of their salary,
working hours, or type of internship."

YOUR EXPECTED OUTPUT:
{"correctness": 0, "coverage": 0, "consistency": 0, "groundedness": 0, "relevance": 1}
'''.strip()


class EvaluationPipeline:
    def __init__(self, system_prompt: str = JUDGE_SYSTEM_PROMPT):
        # Load model ONCE
        self.tokenizer, self.model = load_model()
        self.system_prompt = system_prompt

    def judge_candidate_answer():
        raise NotImplementedError("To be implemented")

        return ask(
            tokenizer=self.tokenizer,
            model=self.model,
            system_prompt=self.system_prompt,
            question=user_query,
            think=False,
        )


def judge_llm(system_prompt: str, user_prompt: str) -> str:
    # Replace this stub with your evaluator API call.
    # Return the evaluator's response as a plain string containing JSON.
    #
    # Example integration shape:
    # response = YOUR_CLIENT.generate(system=system_prompt, prompt=user_prompt)
    # return response.text
    raise NotImplementedError("Connect judge_llm() to your preferred evaluator model.")


def make_user_prompt(row):
    metadata = {
        "id": str(row.get("id", "")),
        "type": row.get("type"),
        "domains": row.get("domains"),
        "primary_domain": row.get("primary_domain"),
        "is_unanswerable": row.get("is_unanswerable", False),
    }
    return f'''
Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

USER INPUT:
{row.get("user_input", "")}

GROUND TRUTH:
{row["reference"]}

CANDIDATE LLM ANSWER:
{row["response"]}

Compare the candidate with the ground truth using the rubric. Return only the required JSON.
'''.strip()


def parse_judge_json(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", raw, flags=re.I)
    result = json.loads(raw)
    score = int(result["score"])
    if not 0 <= score <= 5:
        raise ValueError(f"Judge score must be 0...5, got {score}")
    return result


# TODO (team): TO BE REVISED
def run_evaluation(judge_function, data_df: pd.DataFrame,
                   success_threshold: int = 60, pause_seconds=0):
    results = []
    for _, row in data_df.iterrows():
        raw = judge_function(JUDGE_SYSTEM_PROMPT, make_user_prompt(row=row))
        result = parse_judge_json(raw)
        result["id"] = row["id"]
        # Scale judge score to Y*20/100 from Y/5
        result["judge_score_100"] = result["score"] * 20
        result["success"] = result["judge_score_100"] >= SUCCESS_THRESHOLD
        results.append(result)
        if pause_seconds:
            time.sleep(pause_seconds)
        return pd.DataFrame(results)


# TODO (team): TO BE REVISED
def summarize(eval_frame: pd.DataFrame):
    evaluated = eval_frame.dropna(subset=["judge_score_100"])
    if evaluated.empty:
        print("No judge results found. Run the evaluator first.")
        return
    print(f"Evaluated records: {len(evaluated)}")
    print(f"Mean Embedding Similary (Cosine): {evaluated['embedding_similarity'].mean():.1f}/100")
    print(f"Mean semantic score: {evaluated['judge_score_100'].mean():.1f}/100")
    print(f"Success rate (score >= {SUCCESS_THRESHOLD}): {evaluated['success'].mean()*100:.1f}%")
