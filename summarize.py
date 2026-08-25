from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _client():
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.groq.com/openai/v1")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30,
        max_retries=0,
    )


def _chat_model_name():
    return os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")


sum_prompt = """
Generate a structured summary from the provided medical source (report, paper, or book), strictly adhering to the following categories. The summary should list key information under each category in a concise format: 'CATEGORY_NAME: Key information'. No additional explanations or detailed descriptions are necessary unless directly related to the categories:

ANATOMICAL_STRUCTURE: Mention any anatomical structures specifically discussed.
BODY_FUNCTION: List any body functions highlighted.
BODY_MEASUREMENT: Include normal measurements like blood pressure or temperature.
BM_RESULT: Results of these measurements.
BM_UNIT: Units for each measurement.
BM_VALUE: Values of these measurements.
LABORATORY_DATA: Outline any laboratory tests mentioned.
LAB_RESULT: Outcomes of these tests (e.g., 'increased', 'decreased').
LAB_VALUE: Specific values from the tests.
LAB_UNIT: Units of measurement for these values.
MEDICINE: Name medications discussed.
MED_DOSE, MED_DURATION, MED_FORM, MED_FREQUENCY, MED_ROUTE, MED_STATUS, MED_STRENGTH, MED_UNIT, MED_TOTALDOSE: Provide concise details for each medication attribute.
PROBLEM: Identify any medical conditions or findings.
PROCEDURE: Describe any procedures.
PROCEDURE_RESULT: Outcomes of these procedures.
PROC_METHOD: Methods used.
SEVERITY: Severity of the conditions mentioned.
MEDICAL_DEVICE: List any medical devices used.
SUBSTANCE_ABUSE: Note any substance abuse mentioned.
Each category should be addressed only if relevant to the content of the medical source. Ensure the summary is clear and direct, suitable for quick reference.
"""

def call_openai_api(chunk):
    client = _client()
    response = client.chat.completions.create(
        model=_chat_model_name(),
        messages=[
            {"role": "system", "content": sum_prompt},
            {"role": "user", "content": f" {chunk}"},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
        timeout=45,
    )
    return response.choices[0].message.content


def _local_summary(chunk):
    chunk = chunk.strip()
    if not chunk:
        return ""
    return chunk[:1500]


def split_into_chunks(text, tokens=500):
    model_name = _chat_model_name()
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(' '.join(encoding.decode(words[i:i + tokens])))
    return chunks

def process_chunks(content):
    chunks = split_into_chunks(content)

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        return [_local_summary(chunk) for chunk in chunks]

    # Process chunks in parallel
    try:
        with ThreadPoolExecutor() as executor:
            responses = list(executor.map(call_openai_api, chunks))
        return responses
    except Exception:
        return [_local_summary(chunk) for chunk in chunks]

# Can take up to a few minutes to run depending on the size of your data input
