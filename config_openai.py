import os

from dotenv import dotenv_values, load_dotenv


def load():
    # Loads the .env file and sets variables in this module's global scope.
    load_dotenv()
    values_env_openai = dotenv_values(".env")

    # Set variables for OpenAI/Azure
    global AZURE_OPENAI_API_KEY, endpoint, AZURE_EMBEDDING_DEPLOYMENT
    global deployment_id_gpt4, key, location

    AZURE_OPENAI_API_KEY = values_env_openai["AZURE_OPENAI_API_KEY"]
    endpoint = values_env_openai.get(
        "AZURE_OPENAI_ENDPOINT_URL"
    ) or values_env_openai.get("endpoint")
    AZURE_EMBEDDING_DEPLOYMENT = values_env_openai["AZURE_EMBEDDING_DEPLOYMENT"]

    # For compatibility with your previous code
    deployment_id_gpt4 = values_env_openai.get("deployment_id_gpt4", "gpt-4.1-aias")
    key = values_env_openai.get("key", AZURE_OPENAI_API_KEY)
    location = values_env_openai.get("location", "westeurope")


# Call load() automatically on import
load()
