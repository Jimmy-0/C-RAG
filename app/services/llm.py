from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document

from azure.core.exceptions import HttpResponseError

from config import OpenaiConfig
from app.utils.utils import measure_time
from app.services.NLU import NLU_classification, NLUOutput
from app.utils.prompts import initial_data_chain_prompt, refine_query_prompt, sensitive_word_response
from app.utils.load_data import LoadHRdata
import re

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

class HrTalk:
    def __init__(self, load_data: LoadHRdata):
        self.current_conversation_id = None
        self.conversations_memory = {}
        self.user_personas = {}
        self.load_data = load_data
        self.llm = OpenaiConfig.initail_azurechatai_gpt4o()
        self.output_parser = CommaSeparatedListOutputParser()
        self.data_chain = self._initial_data_chain()
        self.chain_memories = ConversationBufferMemory(input_key="input", return_messages=True)
        refine_query_prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template=refine_query_prompt
        )

        self.conversation_chain = ConversationChain(
            llm=self.llm,
            prompt=refine_query_prompt_template,
            memory=self.chain_memories,
            input_key = 'input',
            output_key="output" 
        )

        

    @measure_time
    def _initial_data_chain(self):
        data_prompt = ChatPromptTemplate.from_messages([
            ("system", initial_data_chain_prompt),
            ("human", "{text}")
        ])
        return data_prompt | self.llm | RunnablePassthrough()
    
    def update_user_persona(self, conversation_id: str, nlu_output: NLUOutput):
        if conversation_id not in self.user_personas:
            self.user_personas[conversation_id] = {
                'intents': set(),
                'entities': set(),
                'keywords': set()
            }
        
        persona = self.user_personas[conversation_id]
        persona['intents'].add(nlu_output.intent.value)
        persona['entities'].update(e.value for e in nlu_output.entities)
        persona['keywords'].update(nlu_output.keywords)
    
    def get_user_persona(self, conversation_id: str) -> str:
        if conversation_id not in self.user_personas:
            return "No persona information available."
        
        persona = self.user_personas[conversation_id]
        return f"""
        User Persona:
        Intents: {', '.join(persona['intents'])}
        Entities: {', '.join(persona['entities'])}
        Keywords: {', '.join(persona['keywords'])}
        """
    
    def extract_content(self, result: Any) -> str:
        if isinstance(result, AIMessage):
            return result.content
        elif isinstance(result, dict):
            return result.get('text', '')
        return str(result)
    
    @measure_time
    def analyze_chain(self, input_: str, nlu_output: NLUOutput, conversation_id: str, context: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        user_persona = self.get_user_persona(conversation_id)
        search_query = input_
        

        if self.chain_memories.buffer:
            print("will refine you  query with history")
            chain_input = {
                "input": input_,
                "history": self.chain_memories.buffer
            }
            search_query = self.conversation_chain.run(chain_input)
            print(f"analyze_chain - refined query: {search_query}")
        else:
            print("This is the first conversation, will not refine.")
            search_query = search_query
        print("="*len(search_query)*2)
        print(search_query)
        print("="*len(search_query)*2)
        search_results = self.load_data.max_marginal_relevance_search(search_query, k=8)

        try:
            with get_openai_callback() as cb:
                result = self.data_chain.invoke({
                    'question': search_query,
                    'text': search_results,
                    'persona': self.chain_memories.buffer,
                    'ori_input': input_,
                })
                cost_detail_list = [{
                    'function': 'data_chain',
                    'model': 'gpt-4o-2024-05-13', 
                    'usage': {
                        'completion_tokens': cb.completion_tokens,
                        'prompt_tokens': cb.prompt_tokens,
                        'total_tokens': cb.total_tokens
                    }
                }]
            
            self.chain_memories.save_context({"input": input_}, {"output": self.extract_content(result)})
            return self.extract_content(result), cost_detail_list
        except Exception as e:
            return f"Error: {str(e)}", None
        
    @measure_time
    def chat_with_follow_up(self, request: dict) -> Tuple[str, List[Dict[str, Any]], str]:
        try:
            input_ = request['message']
            
            self.current_conversation_id = request['current_conversation_id']
            if self.current_conversation_id not in self.conversations_memory:
                self.conversations_memory[self.current_conversation_id] = []
            
            nlu_output = NLU_classification(input_)
            self.update_user_persona(self.current_conversation_id, nlu_output)
            
            result_text, cost_detail_list = self.analyze_chain(input_, nlu_output, self.current_conversation_id)

            self.conversations_memory[self.current_conversation_id].append({
                'human': input_,
                'ai': result_text,
                'nlu': {
                    'intent': nlu_output.intent.value,
                    'entities': [entity.model_dump() for entity in nlu_output.entities],
                    'keywords': nlu_output.keywords
                },
                'timestamp': datetime.now().isoformat()
            })

            # return result_text, cost_detail_list, ""
            return result_text, cost_detail_list

        except HttpResponseError as e:
            error_message = str(e)
            if "content filter" in error_message.lower():
                result_text = sensitive_word_response
            else:
                result_text = "非常抱歉，發生了意外錯誤。請再試一次，如果問題持續存在，請戳右下角的鴨子，聯繫支援團隊。"
            
            print(f"Azure error: {error_message}")
            return result_text, [], error_message

        except Exception as e:
            error_message = str(e)
            result_text = sensitive_word_response
            print(f"Unexpected error in chat_with_follow_up: {error_message}")
            return result_text, [], error_message

def summarize_search_results(search_results: List[Document]) -> str:
    summary = "Search Results Summary:\n"
    for i, doc in enumerate(search_results, 1):
        summary += f"{i}. Source: {doc.metadata.get('source', 'Unknown')}\n"
        summary += f"   Content: {doc.page_content[:100]}...\n\n"
    return summary