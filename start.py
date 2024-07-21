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
from llm_utils import load_and_initialize_llm


PERSIST_DIR = "./storage"
DATA_DIR = "data"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    load_and_initialize_llm()

    index = load_or_create_index(PERSIST_DIR, DATA_DIR)
    
    query_engine = index.as_query_engine()
    chat_with_bot(query_engine)

if __name__ == "__main__":
    main()