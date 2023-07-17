import unittest
import os
import asyncio
import utils
import abkq

class TestABKQ(unittest.TestCase):
    
    def test_get_embeddings_from_text(self):
        text = "This is a sample text for testing."
        embedding = utils.get_embeddings_from_text(text)
        self.assertIsInstance(embedding, dict)
        self.assertIn("vector", embedding)
    
    def test_read_file(self):
        # Create a temporary file for testing
        with open("test_file.txt", "w") as file:
            file.write("This is a test file.")
        
        content = utils.read_file("test_file.txt")
        self.assertEqual(content, "This is a test file.")
        
        # Clean up the temporary file
        os.remove("test_file.txt")
    
    def test_validate_embedding(self):
        valid_embedding = {'vector': [1, 2, 3]}
        invalid_embedding = {'invalid_key': [1, 2, 3]}
        self.assertTrue(utils.validate_embedding(valid_embedding))
        self.assertFalse(utils.validate_embedding(invalid_embedding))
    
    async def async_test_upsert_embeddings_to_pinecone(self):
        embedding = {'vector': [1, 2, 3]}
        book_id = "test_book"
        result = await abkq.upsert_embeddings_to_pinecone(embedding, book_id)
        self.assertTrue(result)
    
    def test_upsert_embeddings_to_pinecone(self):
        # Run the async test function in an event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_test_upsert_embeddings_to_pinecone())
    
    def test_chatbot_query(self):
        user_query = "Tell me about science fiction books."
        response = abkq.chatbot_query(user_query)
        self.assertIsInstance(response, str)
        self.assertTrue(response.startswith("Here are the results:"))

if __name__ == '__main__':
    unittest.main()
