import os
import hashlib
import sqlite3
from typing import List, Optional
from datetime import datetime
import logging
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from config import OpenaiConfig
from utils import measure_time

logging.getLogger("httpx").setLevel(logging.WARNING)

class LoadHRdata:
    def __init__(self, size: int = 1200, overlap: int = 600, data_dir: str = 'data_files', db_path: str = 'hr_data.db'):
        self.hr_rawdata_path = os.path.join(os.path.dirname(__file__), data_dir)
        self.db_path = os.path.join(os.path.dirname(__file__), db_path)
        self.size = size
        self.overlap = overlap
        self.embeddings = self.initial_openaiembed()
        self.init_db()
        logging.basicConfig(level=logging.INFO)
    
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_metadata (
                    filename TEXT PRIMARY KEY,
                    file_hash TEXT,
                    last_processed TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vector_store (
                    id TEXT PRIMARY KEY,
                    data BLOB
                )
            ''')
            conn.commit()

    @staticmethod
    def initial_openaiembed():
        return AzureOpenAIEmbeddings(
            deployment="embeddings",
            model="text-embedding-ada-002",
            azure_endpoint=OpenaiConfig.endpoint,
            openai_api_type="azure",
            openai_api_key=OpenaiConfig.token
        )
    
    def load_pdf(self, file_path):
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    
    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.size,
            chunk_overlap=self.overlap
        )
        return text_splitter.split_documents(documents)
    
    def calculate_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_file_metadata(self, filename: str) -> Optional[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM file_metadata WHERE filename = ?", (filename,))
            return cursor.fetchone()

    def update_file_metadata(self, filename: str, new_hash: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO file_metadata (filename, file_hash, last_processed)
                VALUES (?, ?, ?)
            ''', (filename, new_hash, datetime.now()))
            conn.commit()
            
    def get_vector_store(self) -> Optional[FAISS]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM vector_store WHERE id = 'main'")
            result = cursor.fetchone()
            if result:
                return FAISS.deserialize_from_bytes(
                    pickle.loads(result[0]), 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Add this parameter
                )
        return None
    
    def update_vector_store(self, vector_store: FAISS):
        serialized_data = pickle.dumps(vector_store.serialize_to_bytes())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO vector_store (id, data)
                VALUES (?, ?)
            ''', ('main', serialized_data))
            conn.commit()

    def get_last_update(self) -> Optional[datetime]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(last_processed) FROM file_metadata")
            result = cursor.fetchone()
            return datetime.fromisoformat(result[0]) if result[0] else None
        
    @measure_time
    def pdf_loader(self) -> FAISS:
        all_docs = []
        updated = False

        for file_name in os.listdir(self.hr_rawdata_path):
            if file_name.endswith('.pdf'):
                file_path = os.path.join(self.hr_rawdata_path, file_name)
                file_hash = self.calculate_file_hash(file_path)

                file_meta = self.get_file_metadata(file_name)
                if not file_meta or file_meta[1] != file_hash:
                    logging.info(f"Processing {file_name}")
                    try:
                        doc = self.load_pdf(file_path)
                        new_doc = self.split_documents(doc)
                        all_docs.extend(new_doc)
                        self.update_file_metadata(file_name, file_hash)
                        updated = True
                    except Exception as e:
                        logging.error(f"Error processing {file_name}: {str(e)}")
                else:
                    continue
                    # logging.info(f"Skipping {file_name} (unchanged)")

        vector_store = self.get_vector_store()
        
        if updated:
            for doc in all_docs:
                if 'page' in doc.metadata:
                    doc.metadata['page'] += 1

            if vector_store:
                vector_store.add_documents(all_docs)
            else:
                vector_store = FAISS.from_documents(all_docs, self.embeddings)

            self.update_vector_store(vector_store)
            logging.info("Vector store updated and saved to database.")
        elif vector_store is None:
            vector_store = FAISS.from_documents([], self.embeddings)
            self.update_vector_store(vector_store)
            logging.info("Created an empty vector store.")
        else:
            logging.info("Using existing vector store from database.")

        return vector_store
    
    def get_data_path(self) -> str:
        return self.hr_rawdata_path

    def get_all_file_metadata(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, last_processed FROM file_metadata")
            return cursor.fetchall()














































# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_community.vectorstores import FAISS
# from config import OpenaiConfig
# from utils import measure_time

# class LoadHRdata:
#     def __init__(self, size=1200, overlap=600):
#         self.hr_rawdata_path = './data_files'
#         self.size = size
#         self.overlap = overlap
    
#     def initial_openaiembed(self):
#         return AzureOpenAIEmbeddings(
#             deployment="embeddings",
#             model="text-embedding-ada-002",
#             azure_endpoint=OpenaiConfig.endpoint,
#             openai_api_type="azure",
#             openai_api_key=OpenaiConfig.token
#         )
    
#     @measure_time
#     def pdf_loader(self):
#         all_docs = []
#         for file_name in os.listdir(self.hr_rawdata_path):
#             if file_name.endswith('.pdf'):
#                 file_path = os.path.join(self.hr_rawdata_path, file_name)
#                 loader = PDFPlumberLoader(file_path)
#                 doc = loader.load()
#                 text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=self.size,
#                     chunk_overlap=self.overlap
#                 )
#                 new_doc = text_splitter.split_documents(doc)
#                 all_docs.extend(new_doc)
        
#         for doc in all_docs:
#             if 'page' in doc.metadata:
#                 doc.metadata['page'] += 1
        
#         return FAISS.from_documents(all_docs, self.initial_openaiembed())
