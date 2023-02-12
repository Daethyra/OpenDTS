import re
import requests
import os
import multiprocessing
import time

RATE_LIMIT_SECONDS = 0.33

def generate_word_pattern(word):
    if is_word_valid(word):
        toxicity_score = get_toxicity_score(word)
        if toxicity_score is not None and toxicity_score < 0.7:
            pattern = r"\b" + re.escape(word) + r"\b|\b" + re.escape(re.sub(r"\w", r"[\w@\-]*", word)) + r"\b"
        else:
            pattern = r""
    else:
        pattern = r""
    return pattern

def is_word_valid(word):
    time.sleep(RATE_LIMIT_SECONDS)  # rate limit the API calls
    response = requests.get(f'https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}?key=<your API key>')
    data = response.json()

    if isinstance(data, list):
        # If the API returns a list, the word is valid
        return True
    else:
        # If the API returns a dictionary with a 'meta' key, the word is not valid
        return 'meta' in data.keys()

def get_toxicity_score(word):
    response = requests.post(
        'https://api.openai.com/v1/content/moderation',
        headers={'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}'},
        json={
            'prompt': f'Rate the toxicity of the word "{word}".',
            'max_tokens': 1,
            'n': 1,
            'stop': ['\n']
        }
    )

    if response.status_code == 200:
        data = response.json()
        if len(data['choices']) > 0 and 'text' in data['choices'][0]:
            toxicity_score = float(data['choices'][0]['text'].strip())
            return toxicity_score
    return None

def generate_word_pattern_multiprocess(words):
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_processes) as p:
        patterns = p.map(generate_word_pattern, words)
    return patterns
