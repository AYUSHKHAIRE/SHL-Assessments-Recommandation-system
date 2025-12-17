import os
import google.generativeai as genai
from dotenv import load_dotenv

class GoogleGemini:
    def __init__(self, api_key: str | None = None):
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt
        )
        return response.text