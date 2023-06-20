import requests

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
