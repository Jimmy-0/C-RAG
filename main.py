from load_data import LoadHRdata
from llm import HrTalk
from utils import measure_time
import time

@measure_time
def main():
    start_time = time.time()

    load_data = LoadHRdata(data_dir='data_files', db_path='hr_data.db')
    vector_store = load_data.pdf_loader()
    pdf_loading_time = time.time() - start_time
    print(f"PDF loading and vector store preparation time: {pdf_loading_time:.2f} seconds")

    hr = HrTalk(vector_store)
    hr.initail_azurechatai()
    hr.initial_data_chain()
    hr.initial_word_prompt()

    msg = str('生小孩')
    data = {
        'message': msg
    }
    query_start_time = time.time()
    result, cost_details = hr.chat_with_follow_up(data)
    query_time = time.time() - query_start_time
    
    print(result)
    print("================================")
    print(f"\nQuery execution time: {query_time:.2f} seconds")
    print("Query:", msg)
    print("Response:", result)
    print("Cost Details:", cost_details)
    

if __name__ == '__main__':
    main()