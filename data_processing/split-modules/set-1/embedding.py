import openai
from config import OPENAI_API_KEY, MODEL_ENGINE

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

def generate_embedding(text):
    response = openai.Embedding.create(
        model=MODEL_ENGINE,
        texts=[text]
    )
    return response
