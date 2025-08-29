import os
import glob
import io
import re
import time
import requests
from pypdf import PdfReader, PdfWriter
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.search.documents import SearchClient
from dotenv import load_dotenv, dotenv_values

# Load env/config
load_dotenv()
values_env = dotenv_values(".env")

AZURE_AI_SEARCH_ENDPOINT = f"https://{values_env['searchservice']}.search.windows.net"
AZURE_AI_SEARCH_KEY = values_env['searchkey']
AZURE_STORAGE_NAME = values_env['storageaccount']
AZURE_STORAGE_KEY = values_env['storagekey']
AZURE_BLOB_CONTAINER = values_env['container']
AZURE_OPENAI_ENDPOINT_URL = values_env['endpoint']
AZURE_EMBEDDING_DEPLOYMENT = values_env['AZURE_EMBEDDING_DEPLOYMENT']
AZURE_OPENAI_API_KEY = values_env['AZURE_OPENAI_API_KEY']
DOCS_PATH = values_env['DOCS_PATH']
INDEX_NAME = "docs-vectorized"
CATEGORY = values_env['category']
MAX_SECTION_LENGTH = 1000
SECTION_OVERLAP = 100
SENTENCE_SEARCH_LIMIT = 100
VERBOSE = str(values_env.get('verbose', 'True')).lower() in ['true', '1', 'yes']

def blob_name_from_file_page(filename, page=0):
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
    else:
        return os.path.basename(filename)

def upload_blobs(filename):
    blob_service = BlobServiceClient(account_url=f"https://{AZURE_STORAGE_NAME}.blob.core.windows.net", credential=AZURE_STORAGE_KEY)
    blob_container = blob_service.get_container_client(AZURE_BLOB_CONTAINER)
    if not blob_container.exists():
        blob_container.create_container()

    if os.path.splitext(filename)[1].lower() == ".pdf":
        reader = PdfReader(filename)
        pages = reader.pages
        for i in range(len(pages)):
            blob_name = blob_name_from_file_page(filename, i)
            if VERBOSE: print(f"\tUploading blob for page {i} -> {blob_name}")
            f = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(pages[i])
            writer.write(f)
            f.seek(0)
            blob_container.upload_blob(blob_name, f, overwrite=True)
    else:
        blob_name = blob_name_from_file_page(filename)
        with open(filename, "rb") as data:
            blob_container.upload_blob(blob_name, data, overwrite=True)

def get_document_text(filename):
    offset = 0
    page_map = []
    reader = PdfReader(filename)
    pages = reader.pages
    for page_num, p in enumerate(pages):
        page_text = p.extract_text() or ""
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)
    return page_map

def split_text(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word
        if end < length:
            end += 1

        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, page_map[0][0] if page_map else 0)

        start = end - SECTION_OVERLAP
    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], page_map[0][0] if page_map else 0)

def create_sections(filename, page_map):
    for i, (section, pagenum) in enumerate(split_text(page_map)):
        yield {
            "id": re.sub("[^0-9a-zA-Z_-]", "_", f"{os.path.basename(filename)}-{i}"),
            "content": section,
            "category": CATEGORY,
            "sourcepage": blob_name_from_file_page(filename, pagenum),
            "sourcefile": os.path.basename(filename)
        }

def get_embedding(text):
    url = f"{AZURE_OPENAI_ENDPOINT_URL}openai/deployments/{AZURE_EMBEDDING_DEPLOYMENT}/embeddings?api-version=2024-02-15-preview"
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    embedding = response.json()["data"][0]["embedding"]
    return embedding

def create_vector_index():
    if VERBOSE: print(f"Ensuring vector search index {INDEX_NAME} exists")
    index_client = SearchIndexClient(endpoint=AZURE_AI_SEARCH_ENDPOINT, credential=AzureKeyCredential(AZURE_AI_SEARCH_KEY))
    if INDEX_NAME not in index_client.list_index_names():
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True, facetable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sourcepage", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sourcefile", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=3072,  # for text-embedding-3-large
                vector_search_profile_name="my-vector-profile",
            ),
        ]
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="my-vector-profile",
                    algorithm_configuration_name="my-hnsw-config"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-hnsw-config",
                    kind="hnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ]
        )
        search_index = SearchIndex(
            name=INDEX_NAME,
            fields=fields,
            vector_search=vector_search
        )
        index_client.create_or_update_index(search_index)
        if VERBOSE: print(f"Created search index {INDEX_NAME}.")
    else:
        if VERBOSE: print(f"Search index {INDEX_NAME} already exists.")

def index_sections(sections):
    search_client = SearchClient(endpoint=AZURE_AI_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(AZURE_AI_SEARCH_KEY))
    i = 0
    batch = []
    for s in sections:
        try:
            embedding = get_embedding(s["content"])
            s["embedding"] = embedding
        except Exception as e:
            print(f"Embedding failed for {s['id']}: {e}")
            continue
        batch.append(s)
        i += 1
        if i % 100 == 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            if VERBOSE: print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []
    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        if VERBOSE: print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")

if __name__ == "__main__":
    create_vector_index()
    for filename in glob.glob(DOCS_PATH + "/*.pdf"):
        if VERBOSE: print(f"Processing '{filename}'")
        upload_blobs(filename)
        page_map = get_document_text(filename)
        sections = list(create_sections(filename, page_map))
        index_sections(sections)
