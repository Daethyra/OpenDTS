# This is purely for filtered streaming

import requests
import json
import time
import datetime

def create_headers(bearer_token):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    return headers

def get_existing_rules(headers):
    response = requests.get("https://api.twitter.com/2/tweets/search/stream/rules", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Cannot get rules (HTTP {response.status_code}): {response.text}")
    return response.json()

def rules_are_equal(existing_rules, new_rules):
    existing_rules_set = {(rule['value'], rule['tag']) for rule in existing_rules.get('data', [])}
    new_rules_set = {(rule['value'], rule['tag']) for rule in new_rules}
    return existing_rules_set == new_rules_set

def set_rules(headers, rules):
    payload = {"add": rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            f"Cannot set rules (HTTP {response.status_code}): {response.text}"
        )

def delete_all_rules(headers):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", headers=headers
    )
    if response.status_code != 200:
        raise Exception(
            f"Cannot get rules (HTTP {response.status_code}): {response.text}"
        )
    rules = response.json()
    ids = [rule['id'] for rule in rules.get('data', {})]
    if not ids:
        return None
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            f"Cannot delete rules (HTTP {response.status_code}): {response.text}"
        )

def handle_rate_limit(response):
    if response.status_code == 429:
        reset_time = int(response.headers.get("x-rate-limit-reset"))
        current_time = int(time.time())
        sleep_time = reset_time - current_time + 1
        print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)

def stream_to_file_and_stdout(headers):
    response = requests.get("https://api.twitter.com/2/tweets/search/stream", headers=headers, stream=True)
    if response.status_code != 200:
        raise Exception(f"Cannot get stream (HTTP {response.status_code}): {response.text}")

    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    with open(f"twitter_stream_{timestamp}.txt", "w") as file:
        for r in response.iter_lines():
            if r:
                json_response = json.loads(r)
                print(json.dumps(json_response, indent=4))
                file.write(json.dumps(json_response) + "\n")

def main():
    rules = [
        {
            "value": "(LGBTQIA+ OR transgender OR gay OR lesbian OR bisexual OR queer OR intersex OR asexual OR genderfluid OR nonbinary) -has:links lang:en -is:retweet (context:entities:(sentiment: negative OR sentiment: very_negative))", 
            "tag": "LGBTQIA+"
        },
        {
            "value": "('Donald Trump' OR 'Matt Walsh' OR 'dont tread on me' OR 'MAGA' OR 'Second Amendment' OR 'QAnon' OR 'Proud Boys' OR 'Oath Keepers') -has:links lang:en -is:retweet (context:entities:(sentiment: negative OR sentiment: very_negative))", 
            "tag": "Right-Wing Extremism"
        },
        {
            "value": "('white power' OR 'white pride' OR 'white nationalism' OR 'white supremacy' OR 'Ku Klux Klan' OR 'neo-Nazi') -has:links lang:en -is:retweet (context:entities:(sentiment: positive OR sentiment: very_positive))", 
            "tag": "Religious Extremism"
        }
    ]
    bearer_token = "BEARER_TOKEN"
    headers = create_headers(bearer_token)
    
    # Retrieve existing rules
    existing_rules = get_existing_rules(headers)

    # Check if the existing rules match the new rules
    if not rules_are_equal(existing_rules, rules):
        # Delete all existing rules
        delete_all_rules(headers)
        # Set the new rules
        set_rules(headers, rules)

    # Start the stream
    stream_to_file_and_stdout(headers)

if __name__ == "__main__":
    main()
