import os

import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from azure_openai import *
from config import *
from config_openai import deployment_id_gpt4, endpoint, key

# Now you can use these variables as expected


def create_prompt(context, question):
    """
    Formats the retrieved context and user question into a prompt for the LLM.
    """
    prompt = (
        f"Use the following context to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {question}\n\n"
        f"Answer:"
    )
    return prompt


def generate_answer(conversation):
    """
    Calls the OpenAI chat completion model to generate an answer based on the conversation history.
    """
    import openai

    from config_openai import deployment_id_gpt4, endpoint, key

    client = openai.AzureOpenAI(
        api_key=key, azure_endpoint=endpoint, api_version="2023-05-15"
    )

    response = client.chat.completions.create(
        model=deployment_id_gpt4,
        messages=conversation,
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


st.header("Search Engine - Document")

user_input = st.text_input(
    "Enter your question here:", "What is Diploblastic and Triploblastic Organisation ?"
)

if st.button("Submit"):

    service_name = searchservice
    key = searchkey
    endpoint = f"https://{searchservice}.search.windows.net/"
    index_name = index

    azure_credential = AzureKeyCredential(key)
    search_client = SearchClient(
        endpoint=endpoint, index_name=index_name, credential=azure_credential
    )

    KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT") or "content"
    KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY") or category
    KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE") or "sourcepage"

    # 1. Get embedding for user query
    query_vector = get_query_embedding(user_input)  # Returns List[float]

    # 2. Vector search (ensure `fields` matches your embedding field name in Azure Search index)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=3,
        fields="embedding",  # Change if your vector field has a different name!
        kind="vector",
    )

    results = search_client.search(
        vector_queries=[vector_query],
        select=[KB_FIELDS_SOURCEPAGE, KB_FIELDS_CONTENT, KB_FIELDS_CATEGORY],
        top=3,
    )

    results_list = [
        doc[KB_FIELDS_SOURCEPAGE]
        + ": "
        + doc[KB_FIELDS_CONTENT].replace("\n", "").replace("\r", "")
        for doc in results
    ]
    content = "\n".join(results_list)

    references = [result.split(":")[0] for result in results_list]
    st.markdown("### References:")
    st.write(" , ".join(set(references)))

    conversation = [
        {
            "role": "system",
            "content": "Assistant is a great language model formed by OpenAI.",
        }
    ]
    prompt = create_prompt(content, user_input)
    conversation.append({"role": "assistant", "content": prompt})
    conversation.append({"role": "user", "content": user_input})
    reply = generate_answer(conversation)

    st.markdown("### Answer is:")
    st.write(reply)
