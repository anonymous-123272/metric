"""Extractor and embedder loading, and the cosine between two texts."""

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from metric.scripts.parsing import evidence_text
from metric.scripts.prompts import fill_prompt, prompt_for, system_text

DEFAULT_LLM_MODEL = "Qwen/Qwen3-8B"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_MAX_TOKENS = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.45
SCORE_DECIMALS = 4

loaded_llms = {}
loaded_embedders = {}


def build_messages(method, document, aspect, llm_model):
    """Chat messages for one document. Gemma and DeepSeek get a user message only."""
    content = fill_prompt(prompt_for(method), document, aspect)
    name = llm_model.lower()
    if "gemma" in name or "deepseek" in name:
        return [{"role": "user", "content": content}]
    return [
        {"role": "system", "content": system_text(method)},
        {"role": "user", "content": content},
    ]


def render_prompt(tokenizer, llm_model, messages):
    """Turn chat messages into the string vLLM will generate from."""
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3 chat templates emit a thinking block unless this flag is set.
    if "qwen" in llm_model.lower():
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def get_llm(options):
    """Load and cache the extractor. The cache key is the model id and length."""
    key = (options.llm_model, options.max_model_len, options.gpu_memory_utilization)
    if key not in loaded_llms:
        tokenizer = AutoTokenizer.from_pretrained(options.llm_model, trust_remote_code=True)
        engine = LLM(
            model=options.llm_model,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=options.max_model_len,
            gpu_memory_utilization=options.gpu_memory_utilization,
        )
        loaded_llms[key] = (tokenizer, engine)
    return loaded_llms[key]


def generate_texts(options, prompts_list):
    """Generate one completion per prompt. vLLM continuous-batches the list."""
    engine = get_llm(options)[1]
    sampling = SamplingParams(
        temperature=0,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=options.max_tokens,
    )
    outputs = engine.generate(prompts_list, sampling, use_tqdm=False)
    return [item.outputs[0].text for item in outputs]


def get_embedder(model_id):
    """Load and cache a SentenceTransformer by its full model id."""
    if model_id not in loaded_embedders:
        loaded_embedders[model_id] = SentenceTransformer(model_id, trust_remote_code=True)
    return loaded_embedders[model_id]


def embed_cosine(left, right, embed_model):
    """Return the cosine of two already-extracted texts, rounded to 4 decimals."""
    model = get_embedder(embed_model)
    vectors = model.encode(
        [left, right],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cosine = float(np.dot(vectors[0], vectors[1]))
    return round(cosine, SCORE_DECIMALS)


def extract_documents(documents, aspect_text, options):
    """Return evidence for each document. A blank document is an empty extraction."""
    pending = [index for index, document in enumerate(documents) if document]
    evidence = [""] * len(documents)
    if not pending:
        return evidence
    raws = generate_texts(options, rendered_prompts(documents, aspect_text, options, pending))
    for index, raw in zip(pending, raws):
        evidence[index] = evidence_text(raw)
    return evidence


def rendered_prompts(documents, aspect_text, options, pending):
    """Render one chat prompt per document that has text."""
    tokenizer = get_llm(options)[0]
    rendered = []
    for index in pending:
        messages = build_messages(
            options.method,
            documents[index],
            aspect_text,
            options.llm_model,
        )
        rendered.append(render_prompt(tokenizer, options.llm_model, messages))
    return rendered
