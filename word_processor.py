import re
import openai
import os
import multiprocessing
import time

RATE_LIMIT_SECONDS = 0.33

openai.api_key = os.environ["OPENAI_API_KEY"]

def generate_word_pattern(word):
    if is_word_valid(word):
        toxicity_score = get_toxicity_score(word)
        if toxicity_score is not None:
            pattern = r"\b" + re.escape(word) + r"\b|\b" + re.escape(re.sub(r"\w", r"[\w@\-]*", word)) + r"\b"
        else:
            pattern = r""
    else:
        pattern = r""
    return pattern


def is_word_valid(word):
    time.sleep(RATE_LIMIT_SECONDS)  # rate limit the API calls
    try:
        openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Is {word} a valid word?",
            max_tokens=1,
            n=1,
            stop='\n'
        )
        return True
    except openai.error.OpenAIError:
        return False


def get_toxicity_score(word):
    try:
        response = openai.Completion.create(
            engine="content-filter-alpha-1",
            prompt=f"Rate the toxicity of the word \"{word}\".",
            max_tokens=1,
            n=1,
            stop='\n'
        )

        toxicity_score = float(response.choices[0].text.strip())
        return toxicity_score
    except (openai.error.OpenAIError, ValueError):
        return None


def generate_word_pattern_multiprocess(words):
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_processes) as p:
        patterns = p.map(generate_word_pattern, words)
    return patterns
