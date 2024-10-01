import os
from dotenv import load_dotenv


def load_config():
    if not load_dotenv():
        raise EnvironmentError("Failed to load .env file")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL_NAME")

    if not api_key or not base_url:
        raise EnvironmentError("API key or base URL not found in environment variables")
    return {"model_name": model_name, "api_key": api_key, "base_url": base_url}
