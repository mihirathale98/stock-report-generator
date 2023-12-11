import openai

"""
This is a wrapper for OpenAI API
"""

class OpenAI:
    """Wrapper for OpenAI API"""
    def __init__(self,api_key, model_name='gpt-3.5-turbo'):
        """
        Initialize the API

        Args:
            api_key (str): API key for OpenAI
            model_name (str): Model name from OpenAI

        """
        self.openai_api = openai
        self.openai_api.api_key = api_key
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, model_input, max_new_tokens=512):
        """
        Generate a response
        
        Args:
            model_input (str): Input to the model
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            response (str): Generated response
            
        """
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            temperature=1,
            messages=[
                {"role": "user", "content": model_input}
            ],
            max_tokens=max_new_tokens,
            stop='<|endoftext|>'
        )
        return response.choices[0]["message"]["content"]
