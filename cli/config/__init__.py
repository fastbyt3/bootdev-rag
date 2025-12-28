import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self) -> None:
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
