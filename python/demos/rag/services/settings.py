import os
from dotenv import load_dotenv


class Settings:
    def __init__(self):
        load_dotenv('../../.env')
        self.endpoint = os.getenv('ENDPOINT')
        self.api_key = os.getenv('API_KEY')
        self.api_version = os.getenv('API_VERSION')
        self.gpt_model = os.getenv('GPT_MODEL')
        self.ada_model = os.getenv('ADA_MODEL')
