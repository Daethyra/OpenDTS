import tweepy

# Define a stream listener
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        # Pass the tweet text to the next module for further processing
        from text_processing import process_text
        process_text(status.text)

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)

# Start streaming tweets
def start_streaming():
    myStream.filter(track=['#'])
