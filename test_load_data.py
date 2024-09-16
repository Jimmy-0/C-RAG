import logging
from datetime import datetime
from load_data import LoadHRdata

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_data_loading():
    # Initialize LoadHRdata
    loader = LoadHRdata(data_dir='data_files', db_path='hr_data.db')
    
    # Load PDFs and create/update vector store
    vector_store = loader.pdf_loader()
    
    # Verify vector store
    if vector_store:
        logging.info(f"Vector store created/updated successfully. Index contains {vector_store.index.ntotal} vectors.")
    else:
        logging.error("Failed to create/update vector store.")
        return

    # Check last update time
    last_update = loader.get_last_update()
    logging.info(f"Database last updated: {last_update}")

    # List all processed files
    files = loader.get_all_file_metadata()
    logging.info(f"Processed files ({len(files)}):")
    for file in files:
        logging.info(f"  - {file[0]} (Last processed: {file[1]})")

    # Perform a sample similarity search
    query = "employee benefits"
    results = vector_store.similarity_search(query, k=3)
    logging.info(f"\nSample similarity search for query: '{query}'")
    for i, result in enumerate(results, 1):
        logging.info(f"Result {i}:")
        logging.info(f"  Content: {result.page_content[:100]}...")  # Show first 100 characters
        logging.info(f"  Source: {result.metadata.get('source', 'Unknown')}, Page: {result.metadata.get('page', 'Unknown')}")

if __name__ == "__main__":
    setup_logging()
    verify_data_loading()