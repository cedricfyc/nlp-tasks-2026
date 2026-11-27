"""Hard citations: numbered sources in the prompt and a strict check of every answer.

Each retrieved chunk is shown to the generator as a numbered source [1], [2], ...
The generator must end every factual sentence with the number(s) of the source(s)
that support it. After generation:

  check_citations()     normalises variants ([CONTEXT 2], [Source 2], [2, 3], [2-3])
                        to [2][3], removes numbers that do not belong to a retrieved
                        chunk and lists the factual sentences that have no citation.
  citation_feedback()   correction message for one retry (see pipeline.py).
  finalise_citations()  strict mode drops sentences that are still uncited; an answer
                        left with no valid citation becomes the refusal message.
                        Citations are renumbered 1..m in order of first use and each
                        one is linked to its chunk, document, section, pages and URL.

So an answer never leaves the pipeline with a claim that points to nothing.
"""

import re
from dataclasses import asdict, dataclass, field

import pandas as pd

REFUSAL_MESSAGE = "I don't have enough information to answer that."

# [1]  [1, 3]  [1][3]  [1-3]  [CONTEXT 1]  [Source 2, 4]  (Context 1)
_NUM = r"\d{1,3}"
_KEYWORD = r"(?:context|source|src|doc(?:ument)?|chunk)\s*"
_LIST = rf"(?:{_KEYWORD})?{_NUM}(?:\s*(?:,|;|-|–|and|&)\s*(?:{_KEYWORD})?{_NUM})*"
CITATION_PATTERN = re.compile(
    rf"\[\s*({_LIST})\s*\]"                       # [1], [1, 2], [CONTEXT 1]
    rf"|\(\s*({_KEYWORD}{_NUM}(?:\s*(?:,|;|-|–|and|&)\s*(?:{_KEYWORD})?{_NUM})*)\s*\)",
    flags=re.IGNORECASE,                          # (Context 1) - keyword required
)
_MARKER = re.compile(r"\[(\d+)\]")

# Sentences that only say what the sources do NOT contain need no citation.
_GAP_STATEMENT = re.compile(
    r"don't have enough information|do not have enough information|not enough information"
    r"|\b(?:does|do|did)\s*(?:not|n't)\s+(?:mention|say|state|specify|contain|cover|include"
    r"|provide|address|explain|describe|give)"
    r"|\bnot\s+(?:mentioned|specified|stated|provided|covered|available|included|addressed"
    r"|described|given)\b"
    r"|\bno\s+(?:information|details|mention|data)\b"
    r"|\bcannot\s+(?:be\s+)?(?:determine|determined|answer|answered|confirm|confirmed)\b"
    r"|\bunclear from\b",
    flags=re.IGNORECASE,
)
_REFUSAL = re.compile(r"(?:don't|do not) have enough information", flags=re.IGNORECASE)

# Split after . ! ? when the next sentence starts with a capital, quote, bracket or bold.
# Digits are excluded so "Abs. 2", "p. 3" and "Nr. 5" are not split.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[*A-ZÄÖÜ])")
_LIST_PREFIX = re.compile(r"^(\s*(?:[-*+•]|\d+[.)])\s+)")
MIN_CLAIM_WORDS = 4   # shorter fragments ("Yes.", "In short:") are not treated as claims


# =====================================================================
# Prompt side: numbered sources
# =====================================================================
def format_pages(page_start, page_end) -> str:
    """'3', '3-5' or '' when the chunk has no page numbers."""
    if page_start is None or pd.isna(page_start):
        return ""
    if page_end is None or pd.isna(page_end) or int(page_start) == int(page_end):
        return f"{int(page_start)}"
    return f"{int(page_start)}-{int(page_end)}"


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def source_title(row) -> str:
    """Readable name of the document a chunk comes from."""
    return _text(getattr(row, "source_title", None)) or _text(getattr(row, "filename", None)) \
        or f"chunk {row.chunk_id}"


def build_cited_context(context_df: pd.DataFrame) -> str:
    """Join the chunks as numbered sources: [n] Source / Section / Content."""
    blocks = []
    for number, row in enumerate(context_df.itertuples(), start=1):
        pages = format_pages(row.page_start, row.page_end)
        header = f"[{number}] Source: {source_title(row)}" + (f" (p. {pages})" if pages else "")
        section = _text(getattr(row, "section", None))
        blocks.append(header + (f"\nSection: {section}" if section else "")
                      + f"\nContent:\n{row.text}")
    return "\n\n---\n\n".join(blocks)


# =====================================================================
# Answer side: parse, check, correct
# =====================================================================
@dataclass
class CitedAnswer:
    """Result of the citation check. `answer` is what the user sees."""
    answer: str
    citations: list = field(default_factory=list)      # one dict per cited source, ids 1..m
    refused: bool = False
    refusal_reason: str = ""                           # "model" | "no_valid_citations" | "no_context"
    invalid_ids: list = field(default_factory=list)    # numbers removed: no such source
    uncited_sentences: list = field(default_factory=list)
    dropped_sentences: list = field(default_factory=list)
    retries: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CitationCheck:
    """Intermediate result of check_citations() (original source numbers)."""
    text: str                       # normalised answer, invalid numbers removed
    n_sources: int
    cited_ids: list
    invalid_ids: list
    uncited_sentences: list
    is_refusal: bool

    @property
    def needs_retry(self) -> bool:
        if self.is_refusal:
            return False
        return bool(self.invalid_ids or self.uncited_sentences or not self.cited_ids)


def _expand(numbers: str) -> list[int]:
    """'1, 3-5' -> [1, 3, 4, 5]."""
    numbers = re.sub(_KEYWORD, "", numbers, flags=re.IGNORECASE)
    ids = []
    for part in re.split(r"\s*(?:,|;|and|&)\s*", numbers, flags=re.IGNORECASE):
        bounds = [int(x) for x in re.findall(r"\d+", part)]
        if len(bounds) == 2 and re.search(r"[-–]", part) and bounds[0] <= bounds[1] <= bounds[0] + 20:
            ids.extend(range(bounds[0], bounds[1] + 1))
        else:
            ids.extend(bounds)
    return ids


def normalise_citations(text: str, n_sources: int):
    """Rewrite every citation variant as [n][m]; drop numbers outside 1..n_sources.

    Returns (text, invalid_ids).
    """
    invalid = []

    def replace(match):
        ids = _expand(match.group(1) or match.group(2))
        valid = [i for i in dict.fromkeys(ids) if 1 <= i <= n_sources]
        invalid.extend(i for i in ids if not 1 <= i <= n_sources)
        return "".join(f"[{i}]" for i in valid)

    text = CITATION_PATTERN.sub(replace, text)
    text = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", text)      # [1] [1] -> [1]
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)            # "word [9]." -> "word ."-> "word."
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text, sorted(set(invalid))


def _split_sentences(line: str) -> list[str]:
    """Split one line into sentences; citations after a full stop stay with their sentence."""
    parts = _SENTENCE_SPLIT.split(line)
    sentences = []
    for part in parts:
        leading = re.match(r"^((?:\[\d+\]\s*)+)(.*)$", part)
        if leading and sentences:              # "Fact. [1] Next." -> "Fact. [1]" + "Next."
            sentences[-1] += " " + leading.group(1).strip()
            part = leading.group(2)
        if part.strip():
            sentences.append(part)
    return sentences


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return (stripped.startswith("#")
            or (stripped.endswith(":") and not _MARKER.search(stripped))
            or bool(re.fullmatch(r"\*\*[^*]+\*\*:?", stripped)))


def _needs_citation(sentence: str) -> bool:
    plain = _MARKER.sub("", sentence)
    plain = re.sub(r"[*_`#>]", "", plain).strip()
    if len(re.findall(r"\w+", plain)) < MIN_CLAIM_WORDS:
        return False
    return not _GAP_STATEMENT.search(plain)


def _walk(text: str):
    """Yield (line_index, prefix, sentences) for every non-heading line."""
    for index, line in enumerate(text.split("\n")):
        if not line.strip() or _is_heading(line):
            continue
        prefix_match = _LIST_PREFIX.match(line)
        prefix = prefix_match.group(1) if prefix_match else ""
        yield index, prefix, _split_sentences(line[len(prefix):])


def check_citations(answer: str, n_sources: int) -> CitationCheck:
    """Normalise citations and find invalid numbers and uncited factual sentences."""
    text, invalid = normalise_citations(answer.strip(), n_sources)
    cited = list(dict.fromkeys(int(n) for n in _MARKER.findall(text)))

    uncited = [sentence.strip()
               for _, _, sentences in _walk(text)
               for sentence in sentences
               if not _MARKER.search(sentence) and _needs_citation(sentence)]

    is_refusal = bool(_REFUSAL.search(text)) and not cited
    return CitationCheck(text=text, n_sources=n_sources, cited_ids=cited, invalid_ids=invalid,
                         uncited_sentences=uncited, is_refusal=is_refusal)


def citation_feedback(check: CitationCheck) -> str:
    """Correction message sent back to the generator for one retry."""
    problems = []
    if check.invalid_ids:
        problems.append("You cited source numbers that do not exist: "
                        + ", ".join(f"[{i}]" for i in check.invalid_ids) + ".")
    if not check.cited_ids:
        problems.append("Your answer contains no citations.")
    if check.uncited_sentences:
        listed = "\n".join(f'- "{s}"' for s in check.uncited_sentences[:8])
        problems.append("These sentences have no citation:\n" + listed)

    return (
        "Your answer broke the citation rules.\n" + "\n".join(problems) + "\n\n"
        "Rewrite the complete answer. End every sentence that states a fact with the "
        f"number of the source that supports it, for example [2] or [1][3]. Only the "
        f"numbers 1 to {check.n_sources} exist. Delete every statement that no source "
        f"supports. If no source answers the question, reply exactly: {REFUSAL_MESSAGE}"
    )


def _citation_record(new_id: int, old_id: int, row, snippet_chars: int) -> dict:
    text = _text(row.text)
    rerank_score = getattr(row, "rerank_score", None)
    return {
        "id": new_id,
        "context_id": old_id,
        "chunk_id": int(row.chunk_id),
        "doc_id": str(row.doc_id),
        "source_title": source_title(row),
        "filename": _text(getattr(row, "filename", None)),
        "section": _text(getattr(row, "section", None)) or None,
        "pages": format_pages(row.page_start, row.page_end),
        "url": _text(getattr(row, "download_url", None)) or _text(getattr(row, "source_url", None)) or None,
        "snippet": text if len(text) <= snippet_chars else text[:snippet_chars].rsplit(" ", 1)[0] + " ...",
        "rerank_score": None if rerank_score is None or pd.isna(rerank_score) else float(rerank_score),
    }


def finalise_citations(check: CitationCheck, context_df: pd.DataFrame, strict=True,
                       snippet_chars=300) -> CitedAnswer:
    """Apply the hard rules and return the answer the user sees.

    strict=True removes sentences that are still uncited. An answer with no valid
    citation left (and that is not already a refusal) becomes REFUSAL_MESSAGE.
    """
    if check.is_refusal:
        return CitedAnswer(answer=REFUSAL_MESSAGE, refused=True, refusal_reason="model",
                           invalid_ids=check.invalid_ids)

    text, dropped = check.text, []
    if strict and check.uncited_sentences:
        lines = text.split("\n")
        for index, prefix, sentences in _walk(text):
            kept = []
            for sentence in sentences:
                if not _MARKER.search(sentence) and _needs_citation(sentence):
                    dropped.append(sentence.strip())
                else:
                    kept.append(sentence.strip())
            lines[index] = (prefix + " ".join(kept)) if kept else None
        text = "\n".join(line for line in lines if line is not None)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    order = list(dict.fromkeys(int(n) for n in _MARKER.findall(text)))
    if not order:
        return CitedAnswer(answer=REFUSAL_MESSAGE, refused=True, refusal_reason="no_valid_citations",
                           invalid_ids=check.invalid_ids,
                           uncited_sentences=check.uncited_sentences, dropped_sentences=dropped)

    renumber = {old: new for new, old in enumerate(order, start=1)}
    text = _MARKER.sub(lambda m: f"[{renumber[int(m.group(1))]}]", text)
    rows = list(context_df.itertuples())
    citations = [_citation_record(new, old, rows[old - 1], snippet_chars)
                 for old, new in renumber.items()]

    return CitedAnswer(answer=text, citations=citations, invalid_ids=check.invalid_ids,
                       uncited_sentences=check.uncited_sentences, dropped_sentences=dropped)


def no_context_answer() -> CitedAnswer:
    """Refusal used when retrieval returned nothing."""
    return CitedAnswer(answer=REFUSAL_MESSAGE, refused=True, refusal_reason="no_context")


def format_references(result: CitedAnswer) -> str:
    """Plain-text reference list for the terminal / notebook."""
    if result.refused:
        return f"(refused: {result.refusal_reason})"
    lines = []
    for c in result.citations:
        where = " - ".join(x for x in (c["section"], f"p. {c['pages']}" if c["pages"] else "") if x)
        lines.append(f"[{c['id']}] {c['source_title']}" + (f" ({where})" if where else "")
                     + (f"\n    {c['url']}" if c["url"] else ""))
    if result.dropped_sentences:
        lines.append(f"({len(result.dropped_sentences)} uncited sentence(s) removed)")
    return "\n".join(lines)
