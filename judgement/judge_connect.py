import openai

class GPTJudgeConnection():
    """
    A class to manage interactions with an OpenAI model for judgement tasks.

    This class provides methods to initialize a client, interact with a GPT model, 
    and generate responses based on system and user prompts.

    Attributes:
        model_name (str): The name of the GPT model to use for judgements.
        base_url (str): The base URL for the OpenAI API.

    Methods:
        initialize_client(api_key: str):
            Initializes the OpenAI API client with the specified API key.
        
        create_message(system_prompt: str, user_prompt: str):
            Sends a conversation message to the GPT model and retrieves the response.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
    ):
        self.model_name = model_name
        self.base_url = base_url
    
    def initialize_client(self, api_key: str):
        """
        Initializes the OpenAI client for API interactions.

        Args:
            api_key (str): The API key for authenticating with the OpenAI API.

        Workflow:
            - Creates an `openai.OpenAI` client object.
            - Configures the client with the provided API key and base URL.
        """

        self.client = openai.OpenAI(
            api_key = api_key,
            base_url = self.base_url,
        )
    
    def create_message(self, system_prompt: str, user_prompt: str):
        """
        Generates a response from the GPT model using system and user prompts.

        Args:
            system_prompt (str): The system prompt.
            user_prompt (str): The user prompt.

        Returns:
            str: The generated response content from the GPT model.
        """

        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = [
                {
                    "role": "system", "content": 
                    system_prompt
                },
                {
                    "role": "user", "content":
                    user_prompt
                },
            ],
            temperature = 0.7
        )

        return response.choices[0].message.content