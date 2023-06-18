from utils import create_headers
from rules import get_existing_rules, rules_are_equal, set_rules, delete_all_rules
from stream import stream_to_file_and_stdout
from config import TWITTER_BEARER_TOKEN

def main():
    rules = [
        # ...
    ]
    headers = create_headers(TWITTER_BEARER_TOKEN)
    
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
