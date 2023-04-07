from data_collection import collect_twitter_data, collect_news_data
from data_processing import process_twitter_data, process_news_data
from message_queue import receive_messages
from data_storage import save_data


def get_user_input():
    query = input("Enter the keywords to search for: ")
    return query

def main():
   # Get User's input
    query = get_user_input()
   # Collect and process data
    twitter_data = collect_twitter_data(query)
    process_twitter_data(twitter_data)

    news_data = collect_news_data(query, '2020-01-01', '2023-12-31')
    process_news_data(news_data)

    # Consume processed messages from the queue
    def handle_message(message):
        save_data(message['source'], message['text'], message['sentiment'])

    receive_messages('data_processing', handle_message)

if __name__ == '__main__':
    main()
