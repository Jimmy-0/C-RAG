import os
import pandas as pd
from langchain.schema import Document

class CSVLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load(self):
        print("loading .. OK ")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file '{self.file_path}' does not exist.")
        if not self.file_path.lower().endswith('.csv'):
            raise ValueError(f"The file '{self.file_path}' is not a CSV file.")
        
        try:
            # Read the CSV file using pandas
            df = pd.read_csv(self.file_path, comment='#', skipinitialspace=True)
            
            # Convert DataFrame to list of dictionaries
            data = df.to_dict('records')
            documents = []
            for i, item in enumerate(data):
                # Convert all values to strings to ensure compatibility
                content = "\n".join(f"{k}: {v}" for k, v in item.items())
                metadata = {
                    "source": self.file_path,
                    "row": i,
                    "source_type": "csv"
                }
                documents.append(Document(page_content=content, metadata=metadata))
            
            return documents
        except Exception as e:
            print(f"Error details: {str(e)}")
            raise Exception(f"Error loading CSV: {str(e)}")