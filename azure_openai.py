from openai import AzureOpenAI

from config_openai import *

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=AZURE_OPENAI_API_KEY,  # from .env!
    api_version="2023-05-15",
)


def get_query_embedding(text):
    embedding_response = client.embeddings.create(
        input=[text], model=AZURE_EMBEDDING_DEPLOYMENT  # from .env!
    )
    return embedding_response.data[0].embedding
