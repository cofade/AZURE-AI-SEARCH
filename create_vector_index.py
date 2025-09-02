from dotenv import load_dotenv, dotenv_values
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
import os

# Load environment variables
load_dotenv()
values_env = dotenv_values(".env")

# Azure Search config
searchservice = values_env['searchservice']
searchkey = values_env['searchkey']
index_name = "docs-vectorized"  # Use a new index name for vectors

# Optional: other metadata fields from your config
category = values_env['category']

# Connect to Azure AI Search
endpoint = f"https://{searchservice}.search.windows.net"
credential = AzureKeyCredential(searchkey)

# Define the index fields
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

# Define the vector search configuration
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

# Create the index object
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search
)

# Create the index in Azure AI Search
client = SearchIndexClient(endpoint, credential)
result = client.create_or_update_index(index)
print(f"Index '{result.name}' created or updated.")
