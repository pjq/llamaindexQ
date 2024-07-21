import os
import json
import logging
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

CONFIG_PATH = 'config.json'
PERSIST_DIR = "./storage"
DATA_DIR = "data"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def initialize_llm(settings):
    logging.info("Initializing LLM with provided settings")
    llm_settings = settings['llm']
    Settings.llm = OpenAI(
        temperature=llm_settings['temperature'], 
        model=llm_settings['model'], 
        api_key=llm_settings['api_key'], 
        api_base=llm_settings['api_base']
    )

def create_index(data_dir, persist_dir):
    logging.info(f"Creating index from documents in {data_dir}")
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    logging.info(f"Index created and persisted to {persist_dir}")
    return index

def load_or_create_index(persist_dir, data_dir):
    if not os.path.exists(persist_dir):
        logging.info(f"No existing storage found at {persist_dir}. Creating new index.")
        return create_index(data_dir, persist_dir)
    else:
        logging.info(f"Loading existing index from {persist_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)

def chat_with_bot(query_engine):
    logging.info("ChatBot is ready for interaction")
    print("ChatBot is ready! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            logging.info("Chat session ended by user")
            print("ChatBot: Goodbye!")
            break
        logging.info(f"User query: {user_input}")
        response = query_engine.query(user_input)
        logging.info(f"ChatBot response: {response}")
        print(f"ChatBot: {response}")

def main():
    logging.info("Starting ChatBot application")
    config = load_config(CONFIG_PATH)
    initialize_llm(config['Settings'])
    
    index = load_or_create_index(PERSIST_DIR, DATA_DIR)
    
    query_engine = index.as_query_engine()
    chat_with_bot(query_engine)

if __name__ == "__main__":
    main()