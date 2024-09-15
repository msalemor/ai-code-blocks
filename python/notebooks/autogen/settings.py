import os
from dotenv import load_dotenv


class Settings:
    def __init__(self) -> None:
        load_dotenv("../../.env")
        self.model = os.getenv("GPT_MODEL")
        self.endpoint=os.getenv("ENDPOINT")
        self.api_key=os.getenv("API_KEY")
        self.api_version=os.getenv("API_VERSION")