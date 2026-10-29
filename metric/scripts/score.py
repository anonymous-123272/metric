"""Pair score and command-line entry."""

import argparse
import sys
from dataclasses import dataclass

from metric.scripts.modeling import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_TOKENS,
    embed_cosine,
    extract_documents,
)
from metric.scripts.prompts import METHOD_KEYS


@dataclass
class MetricOptions:
    """Models and extraction method for one similarity call."""

    method: str = "single"
    llm_model: str = DEFAULT_LLM_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    max_tokens: int = DEFAULT_MAX_TOKENS
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION


@dataclass
class ScoreResult:
    """Cosine score, the two evidence strings, and a note when a side is missing."""

    score: float
    left_evidence: str
    right_evidence: str
    missing: str = ""


def missing_evidence_note(left_present, right_present):
    """Name the document side whose extraction is empty. Empty when both sides have text."""
    if left_present and right_present:
        return ""
    if not left_present and not right_present:
        return "Not Found Evidence in A and B"
    if not left_present:
        return "Not Found Evidence in A"
    return "Not Found Evidence in B"


def result_with_missing(score_value, left_evidence, right_evidence):
    """Fill empty evidence with the missing-side message and keep the cosine."""
    note = missing_evidence_note(bool(left_evidence), bool(right_evidence))
    if not left_evidence:
        left_evidence = "Not Found Evidence in A"
    if not right_evidence:
        right_evidence = "Not Found Evidence in B"
    return ScoreResult(score_value, left_evidence, right_evidence, note)


def validate_options(options):
    """Raise when the method name or memory settings cannot be used."""
    if options.method not in METHOD_KEYS:
        raise ValueError("method must be one of: %s" % ", ".join(sorted(METHOD_KEYS)))
    if not options.llm_model.strip() or not options.embed_model.strip():
        raise ValueError("llm_model and embed_model must be non-empty model ids")
    if options.max_model_len < 1 or options.max_tokens < 1:
        raise ValueError("max_model_len and max_tokens must be >= 1")
    if not 0 < options.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")


def score_pair(doc1, doc2, aspect, options=None):
    """Extract both sides and return the cosine plus the two evidence strings.

    doc1 and doc2 are the raw documents. aspect is the condition. options
    selects the prompt and the two model ids. The score is 0 when either
    extraction is empty, and missing names side A, side B, or both.
    """
    options = options or MetricOptions()
    validate_options(options)
    aspect_text = str(aspect).strip()
    if not aspect_text:
        raise ValueError("aspect must be non-empty")
    left_doc = "" if doc1 is None else str(doc1).strip()
    right_doc = "" if doc2 is None else str(doc2).strip()
    left_evidence, right_evidence = extract_documents(
        [left_doc, right_doc], aspect_text, options
    )
    if not left_evidence or not right_evidence:
        return result_with_missing(0.0, left_evidence, right_evidence)
    cosine = embed_cosine(left_evidence, right_evidence, options.embed_model)
    return ScoreResult(cosine, left_evidence, right_evidence, "")


def score(doc1, doc2, aspect, options=None):
    """Return the aspect similarity of two documents as a float."""
    return score_pair(doc1, doc2, aspect, options).score


def read_text_argument(text, path):
    """Use a file when one is given, otherwise the inline string."""
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return text or ""


def parse_args():
    """CLI flags for one aspect-conditioned similarity score."""
    parser = argparse.ArgumentParser(
        description="Score two documents on one aspect with retrieve-then-embed."
    )
    parser.add_argument("--doc1", default="", help="First document text")
    parser.add_argument("--doc2", default="", help="Second document text")
    parser.add_argument("--doc1-file", default="", help="Read the first document from this file")
    parser.add_argument("--doc2-file", default="", help="Read the second document from this file")
    parser.add_argument("--aspect", required=True, help="Aspect that conditions the score")
    parser.add_argument(
        "--method",
        default="single",
        choices=sorted(METHOD_KEYS),
        help="Extraction style (default: single)",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Extractor model id (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Any SentenceTransformer model id (default: Qwen/Qwen3-Embedding-8B)",
    )
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="GPU share reserved for the extractor so the embedder can load afterward",
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print the extracted texts to stderr",
    )
    return parser, parser.parse_args()


def main():
    """Read one pair from the command line and print the similarity."""
    parser, args = parse_args()
    doc1 = read_text_argument(args.doc1, args.doc1_file)
    doc2 = read_text_argument(args.doc2, args.doc2_file)
    if not doc1.strip() or not doc2.strip() or not (args.aspect or "").strip():
        parser.error("Provide both documents and an aspect")
    options = MetricOptions(
        method=args.method,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    result = score_pair(doc1, doc2, args.aspect, options)
    print(result.score)
    if result.missing:
        print(result.missing)
    if args.show_evidence:
        print("A:", result.left_evidence, file=sys.stderr)
        print("B:", result.right_evidence, file=sys.stderr)


if __name__ == "__main__":
    main()
