# AspectSim

Aspect-conditioned document similarity Metric. It assigns a similarity of the two documents with respect to a aspect provided in natural langauge format. The metric follows two steps:

1. An LLM extracts the evidence in each document that pertains to the aspect.
2. Then an embedding model encodes those evidence and the cosine similarity between the resulting vectors is the score.

Quick Python:

```python
from metric.score import score_pair
result = score_pair(doc_a, doc_b, "aspect text")
print(result.score)  # cosine (4 decimals)
```

Quick CLI:

```bash
python -m metric.score --doc1 "..." --doc2 "..." --aspect "aspect" --show-evidence
```

Examples live in `metric/examples/`.

## Python

```python
from metric.score import score_pair

doc1 = "At least 30 people have been killed in the collapse of a gold mine in northeastern Afghanistan."
doc2 = "Villagers dug a makeshift shaft to hunt for gold and were caught when it collapsed."

result = score_pair(doc1, doc2, "Gold mine collapse incident")
print(result.score)
print("A:", result.left_evidence)
print("B:", result.right_evidence)
```

`result.score` is the cosine, rounded to 4 decimals. `left_evidence` is document A and `right_evidence` is document B.

`MetricOptions` changes the extraction style or the model ids. `embed_model` is any SentenceTransformer id.

```python
from metric.score import MetricOptions, score_pair

options = MetricOptions(
    method="single",
    llm_model="Qwen/Qwen3-4B",
    embed_model="google/embeddinggemma-300m",
)
result = score_pair(doc1, doc2, "Gold mine collapse incident", options)
```

`method` is `single`, `multi`, or `summarize`. `single` keeps one sentence, `multi` keeps every relevant sentence, and `summarize` writes a short aspect summary. The default method is `single`.

## Command line

```bash
python -m metric.score \
  --doc1 "At least 30 people have been killed in the collapse of a gold mine in northeastern Afghanistan." \
  --doc2 "Villagers dug a makeshift shaft to hunt for gold and were caught when it collapsed." \
  --aspect "Gold mine collapse incident" \
  --show-evidence
```

The score is printed to stdout. With `--show-evidence`, the two evidence texts are printed to stderr. `--doc1-file` and `--doc2-file` read each document from a text file. `--method`, `--llm-model`, and `--embed-model` match the Python options.

## Missing evidence

When a document does not discuss the aspect, that side is replaced and the score is `0.0`.

| Case       | `left_evidence`       | `right_evidence`      | `missing`                   |
| ---------- | ----------------------- | ----------------------- | ----------------------------- |
| A empty    | Not Found Evidence in A | extracted text from B   | Not Found Evidence in A       |
| B empty    | extracted text from A   | Not Found Evidence in B | Not Found Evidence in B       |
| Both empty | Not Found Evidence in A | Not Found Evidence in B | Not Found Evidence in A and B |

`missing` is an empty string when both sides have evidence.

## Defaults

- method: `single` (default extraction style)
- llm_model: `Qwen/Qwen3-8B`
- embed_model: `Qwen/Qwen3-Embedding-8B`
- max_model_len: `16384`
- max_tokens: `2048`
- gpu_memory_utilization: `0.45`
- score: cosine rounded to 4 decimals
- CLI: use `--show-evidence` to print extracted texts to stderr


