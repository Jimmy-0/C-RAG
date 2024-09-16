import os
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from config import OpenaiConfig
from utils import measure_time

class HrTalk:
    def __init__(self, db):
        self.db = db
        self.word_chain = None
        self.data_chain = None
        self.output_parser = CommaSeparatedListOutputParser()
        self.llm = None
        self.memory = None

    @measure_time
    def initail_azurechatai(self):
        os.environ["AZURE_OPENAI_ENDPOINT"] = OpenaiConfig.endpoint
        os.environ["AZURE_OPENAI_API_KEY"] = OpenaiConfig.token
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt4o",
            api_version="2024-05-01-preview",
            model="gpt-4o",
            model_version="2024-05-13",
            temperature=0
        )
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=1000)

    @measure_time
    def initial_word_prompt(self):
        word_prompt = ChatPromptTemplate.from_messages([
            ("human", "你是公司員工，如果{input}與人資或資料庫有關，從資料庫中提取最相關的20個文檔，**從'考績、津貼、福利、薪資、保險'的角度，聯想出2個員工最關心的關鍵詞**。僅依據我提供的資料庫進行聯想，此關鍵詞必須聚焦於一般員工對人資提問：**{input}**。此關鍵詞必須要符合**廣泛性、多樣性和變化性**，**必須包含{input}相關的完整內容**。此關鍵詞的回覆必須穩定，**相同的提問應得到一致的回覆**。僅回覆聯想問題中最重要、關鍵詞, 並以半形逗號與空格來分隔。不要加入其他內容")
        ])
        self.word_chain = LLMChain(llm=self.llm, prompt=word_prompt)

    @measure_time
    def initial_data_chain(self):
        data_prompt = ChatPromptTemplate.from_messages([
            ("system", "你現在是一位專業、溫柔、體貼員工且熟悉人資規章的國泰健康管理顧問公司人力資源管理師，你會以「關懷員工」的角度回答問題，'回覆的風格必須要像是跟真人在「對話」'。首先判斷如果{input}是否與知識庫，如果沒有，或者{input}不屬於人資專業，說明你拒絕回答的理由，並結束對話。, 如果{input}與知識庫有關，再從知識庫中主要聚焦回答與{input}的問題與建議作法，再來補充與{output}有關的問題, '不需要重複用戶問的問題', 回覆的內容要確保「完整性、具體性、明確性」，如果是違反公司規範，就要用「語氣強硬」「嚴正警告」的態度告訴員工不可以做。**若在{text}中發現，有因為{input}而影響'考績、津貼、薪酬、福利、保險'，就需要優先告知員工具體的影響程度、並給出詳細的數值**, 如果從知識庫有{input}的申請流程，就要告訴員工'詳盡的申請流程', 如果知識庫中提到有路徑可以查詢{input}相關資訊，就要告訴員工'詳盡的查詢方式', 不允許模糊的回覆、不允許有錯誤的回覆、不允許與{input}無關的回覆，僅以提供的文件資料為回答的參考依據**知識來源取用的優先序如下：1. 首先使用「_考勤及團險FAQ.pdf」2. 如果上述資料來源無法提供足夠信息，則使用「_【公司規定】」3. 最後，若前述資料仍不足，才使用「_【法令規定】」**, 如果找不到明確的答案，就回答不知道, 必須在內文中明確的提供知識來源的檔案名稱、出自於第幾條規範、出自於第幾page，**知識來源的格式範例必須如下：<檔案名稱>第N頁,第N條**。 允許資料出自於多個文件, 必須確保知識來源的正確性與完整性, 必須總結出你建議的做法給員工, '你要確實遵照我所有的prompt指令，不然會被懲罰', 在最後提供'知識來源'的file_path'，產出的格式必須讓用戶可以點擊連結，並打開對應的知識文檔, **在生成回覆之前，請進行多層次的內部驗證，確保所有細節準確無誤，**, 以繁體中文輸出"),
            ("human", "{text}")
        ])
        self.data_chain = LLMChain(llm=self.llm, prompt=data_prompt)

    @measure_time
    def analyze_chain(self, input_, context=None):
        if context:
            input_ = f"User_Query: {input_} \n History_Chat: {context}"

        search_results = self.db.max_marginal_relevance_search(input_)

        with get_openai_callback() as cb:
            word = self.word_chain({'input': input_})
            word_chain_cost_detail = {
                'function': 'word_chain',
                'model': 'gpt-4o-2024-05-13',
                'usage': {
                    'completion_tokens': cb.completion_tokens,
                    'prompt_tokens': cb.prompt_tokens,
                    'total_tokens': cb.total_tokens
                }
            }

        word_list = self.output_parser.parse(word['text'])

        for i in word_list:
            search_results += self.db.max_marginal_relevance_search(i, k=5)
            
        word_list.append(input_)
        
        try:
            with get_openai_callback() as cb:
                result = self.data_chain({'input': input_, 'output': word_list, 'text': search_results})
                cost_detail_list = [
                    word_chain_cost_detail,
                    {
                        'function': 'data_chain',
                        'model': 'gpt-4o-2024-05-13',
                        'usage': {
                            'completion_tokens': cb.completion_tokens,
                            'prompt_tokens': cb.prompt_tokens,
                            'total_tokens': cb.total_tokens
                        }
                    }
                ]

            return result['text'], cost_detail_list
        
        except Exception as e:
            generated_text = f"Error: {str(e)}"
            return generated_text, None

    @measure_time
    def chat_with_follow_up(self, request: dict):
        input_ = request['message']
        previous_conversation = self.memory.load_memory_variables({})

        if 'history' in previous_conversation:
            context = previous_conversation['history']
        else:
            context = None

        result_text, cost_detail_list = self.analyze_chain(input_, context)

        self.memory.save_context({'input': input_}, {'output': result_text})
        return result_text, cost_detail_list
