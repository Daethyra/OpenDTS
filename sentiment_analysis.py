import openai

def analyze_sentiment(text, temperature=0.5, max_tokens=50):
    """
    Analyze the sentiment of the given text using ChatGPT3.5-turbo.

    :param text: The input text to analyze.
    :param temperature: The sampling temperature for the model (default: 0.5).
    :param max_tokens: The maximum number of tokens in the response (default: 50).
    :return: The model's response as a string.
    """

    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=f"Analyze the sentiment of the following text: {text}",
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            echo=False
        )

        sentiment = response.choices[0].text.strip()
        return sentiment

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return None
