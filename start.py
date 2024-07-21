import os.path
from llama_index.llms.openai import OpenAI
import json

from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Load configuration from config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

llm_settings = config['Settings']['llm']

Settings.llm = OpenAI(
    temperature=llm_settings['temperature'], 
    model=llm_settings['model'], 
    api_key=llm_settings['api_key'], 
    api_base=llm_settings['api_base']
)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("Summary of the doc?")
print(response)