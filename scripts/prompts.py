"""Single, multi, and summarize extraction prompts."""

METHOD_KEYS = {
    "single": "single_extraction",
    "multi": "multi_extraction",
    "summarize": "summarize",
}

SINGLE_EXTRACTION = """You are a highly intelligent system designed to identify and extract relevant sentence from a document based on a specific aspect.
    You will first determine if the document discusses the given aspect, and then extract the relevant sentence that directly talk about the aspect in the document.

    To complete your task follow these steps carefully:

    Step 1: Does the document discuss the aspect '__ASPECT__'? Answer 'Yes' or 'No'. If 'Yes', proceed to Step 2. If 'No', return an empty list for 'narrative_1'.

    Step 2: If the answer in Step 1 is 'Yes', extract the **relevant key sentence**  from the document that directly discuss the aspect.


    Your output must follow these strict rules:
    - Output a single, valid JSON object.
    - The JSON object must contain two keys: 'discusses_aspect' and 'narrative_1'.
    - The value of 'discusses_aspect' must be either 'Yes' or 'No'.
    - The value of 'narrative_1' should be a sentence that directly discuss the aspect if 'discusses_aspect' is 'Yes'. If 'discusses_aspect' is 'No', 'narrative_1' must be an empty list.


    ###Expected Output format:###

    {"discusses_aspect": "Yes", "narrative_1": ["Matching sentence 1."]}
    or
    {"discusses_aspect": "No", "narrative_1": []}

    Additional instructions:"
    - Do not include any explanations, comments, or text outside the JSON object."
    - Ensure that sentences in the output are complete and well-formed."
    - Do not provide partial JSON responses."


    Now, analyze the following document: __DOCUMENT__, and determine if it discusses the aspect: __ASPECT__. If yes, then extract the relevant sentence that talk about the aspect.
"""

MULTI_EXTRACTION = """You are a highly intelligent system designed to identify and extract relevant sentences from a document based on a specific aspect.
You will first determine if the document discusses the given aspect, and then extract the relevant sentences that talk about the aspect in the document.

    To complete your task follow these steps carefully:

    Step 1: Does the document discuss the aspect '__ASPECT__'? Answer 'Yes' or 'No'. If 'Yes', proceed to Step 2. If 'No', return an empty list for 'narrative_1'.

    Step 2: If the answer in Step 1 is 'Yes', extract **all relevant sentences** from the document that discuss the aspect.


    Your output must follow these strict rules:
    - Output a single, valid JSON object.
    - The JSON object must contain two keys: 'discusses_aspect' and 'narrative_1'.
    - The value of 'discusses_aspect' must be either 'Yes' or 'No'.
    - The value of 'narrative_1' must be a list of sentences that discuss the aspect if 'discusses_aspect' is 'Yes'. If 'discusses_aspect' is 'No', 'narrative_1' must be an empty list.

    Expected format:
    {"discusses_aspect": "Yes", "narrative_1": ["Matching sentence 1.", "Matching sentence 2.",....]}
    or
    {"discusses_aspect": "No", "narrative_1": []}

    Additional instructions:"
    - Do not include any explanations, comments, or text outside the JSON object."
    - Ensure that sentences in the output are complete and well-formed."
    - Do not provide partial JSON responses."

    Now, analyze the following document: __DOCUMENT__, and determine if it discusses the aspect: __ASPECT__. If yes, then extract the relevant sentences that talk about the aspect.
"""

SUMMARIZE = """You are a highly intelligent system designed to identify and summarize the information in a document that is relevant to a specific aspect.
You will first determine if the document discusses the given aspect, and then generate a concise summary (2–3 sentences) that captures how the document addresses that aspect.

To complete your task follow these steps carefully:

Step 1: Does the document discuss the aspect '__ASPECT__'? Answer 'Yes' or 'No'.
If 'Yes', proceed to Step 2.
If 'No', return an empty list for 'narrative_1'.

Step 2: If the answer in Step 1 is 'Yes', write a concise summary (2–4 sentences) that describes how the document discusses the aspect.
The summary must focus only on the information directly related to the aspect, ignoring unrelated details.

Your output must follow these strict rules:
- Output a single, valid JSON object.
- The JSON object must contain two keys: 'discusses_aspect' and 'narrative_1'.
- The value of 'discusses_aspect' must be either 'Yes' or 'No'.
- The value of 'narrative_1' should be a list containing the summarized text if 'discusses_aspect' is 'Yes'.
If 'discusses_aspect' is 'No', 'narrative_1' must be an empty list.

### Expected Output Format ###
{"discusses_aspect": "Yes", "narrative_1": ["Aspect-specific summary here."]}
or
{"discusses_aspect": "No", "narrative_1": []}

Additional instructions:
- Do not include any explanations, comments, or text outside the JSON object.
- Ensure that the summary sentences are complete, well-formed, and aspect-focused.
- Do not provide partial JSON responses.
- Keep the summary short and factual (2–4 sentences maximum).

Now, analyze the following document: __DOCUMENT__, and determine if it discusses the aspect: __ASPECT__.
If yes, generate the aspect-based summary as described above.
"""

PROMPTS = {
    "single_extraction": SINGLE_EXTRACTION,
    "multi_extraction": MULTI_EXTRACTION,
    "summarize": SUMMARIZE,
}


def prompt_for(method):
    """Return the template for single, multi, or summarize."""
    if method not in METHOD_KEYS:
        raise ValueError("method must be one of: %s" % ", ".join(sorted(METHOD_KEYS)))
    return PROMPTS[METHOD_KEYS[method]]


def fill_prompt(template, document, aspect):
    """Insert the document and aspect without interpreting other braces."""
    return template.replace("__DOCUMENT__", document).replace("__ASPECT__", aspect)


def system_text(method):
    """System line that matches the extraction style."""
    if method == "summarize":
        return "You are a precise and efficient aspect based summarization model."
    return "You are a precise and efficient extraction model."
