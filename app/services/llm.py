import os
import sys
import json
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Optional, Any, Tuple

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser

from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback

from config import OpenaiConfig
from app.utils.utils import measure_time
from app.services.NLU import NLU_classification, NLUOutput
from app.utils.prompts import initial_data_chain_prompt,refine_query_prompt
from app.utils.load_data import LoadHRdata
from azure.core.exceptions import HttpResponseError


class SimpleAnalyzeChain:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(template=refine_query_prompt, input_variables=["input", "user_persona", "history"])
        self.chain = LLMChain(llm=llm, prompt=self.prompt)

    def __call__(self, **kwargs):
        return self.chain.run(**kwargs)


class HrTalk:
    def __init__(self, load_data: LoadHRdata):
        self.current_conversation_id = None
        self.conversations_memory = {}
        self.user_personas = {}
        self.load_data = load_data
        self.llm = OpenaiConfig.initail_azurechatai_gpt4o()
        self.output_parser = CommaSeparatedListOutputParser()
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=1000)
        self.data_chain = self._initial_data_chain()
        self.simple_analyze_chain = SimpleAnalyzeChain(self.llm)
        
    @measure_time
    def _initial_data_chain(self):
        data_prompt = ChatPromptTemplate.from_messages([
            ("system", initial_data_chain_prompt),
            ("human", "{text}")
        ])
        # self.data_chain = data_prompt | self.llm | RunnablePassthrough()
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
        
        # print(f"Updated persona: {persona}")
        # print("=================update_user_persona - PERSONA=================")
        # print(f"""
        #     User Persona:
        #     Intents: {', '.join(persona['intents'])}
        #     Entities: {', '.join(persona['entities'])}
        #     Keywords: {', '.join(persona['keywords'])}
        # """)
        # print("=================update_user_persona - PERSONA=================") 

    
    def get_user_persona(self, conversation_id: str) -> str:
        """Get formatted user persona string"""
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
        """Extract content from various result types"""
        if isinstance(result, AIMessage):
            return result.content
        elif isinstance(result, dict):
            return result.get('text', '')
        return str(result)

    @measure_time
    def analyze_chain(self, input_: str, nlu_output: NLUOutput, conversation_id: str, context: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:

        user_persona = self.get_user_persona(conversation_id)
        

        refined_query = self.simple_analyze_chain(
            input=input_,
            history=context,
            user_persona=user_persona
        )
        
        search_results = self.load_data.max_marginal_relevance_search(input_)
    

        try:



            with get_openai_callback() as cb:
                result = self.data_chain.invoke({
                    'input': input_,
                    'text': search_results,
                    'user_persona': user_persona,
                    'history': context
                })
                cost_detail_list = [
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

            return self.extract_content(result), cost_detail_list
        
        except Exception as e:
            generated_text = f"Error: {str(e)}"
            return generated_text, None

    @measure_time
    def chat_with_follow_up(self, request: dict) -> Tuple[str, List[Dict[str, Any]], str]:
        """Main chat function with follow-up"""
        try:
            input_ = request['message']
            self.current_conversation_id = request['current_conversation_id']
            if self.current_conversation_id not in self.conversations_memory:
                self.conversations_memory[self.current_conversation_id] = []

            conversation_id = self.current_conversation_id
            context = self.conversations_memory[conversation_id][-1] if self.conversations_memory[conversation_id] else None

            nlu_output = NLU_classification(input_)
                        
            self.update_user_persona(conversation_id, nlu_output)

            result_text, cost_detail_list = self.analyze_chain(input_, nlu_output, conversation_id, context)
            
            self.conversations_memory[conversation_id].append({
                'human': input_,
                'ai': result_text,
                'nlu': {
                    'intent': nlu_output.intent.value,
                    'entities': [entity.model_dump() for entity in nlu_output.entities],
                    'keywords': nlu_output.keywords
                },
                'timestamp': datetime.now().isoformat()
            })

            return result_text, cost_detail_list,  # Empty string for error message

        except HttpResponseError as e:
            error_message = str(e)
            if "content filter" in error_message.lower():
                result_text = """
                😔非常抱歉，您的查詢內容觸發了我們的內容過濾系統。為了確保對話的安全性和適當性，我無法直接回應您的問題。請您重新表述您的問題，避免使用可能引起敏感內容警報的字詞。如果您有任何疑問或需要協助，請隨時告訴我，我會盡力為您提供適當的幫助。謝謝您的理解與配合。
                """
            else:
                result_text = "非常抱歉，發生了意外錯誤。請再試一次，如果問題持續存在，請戳右下角的鴨子，聯繫支援團隊。"
            
            print(f"Azure error: {error_message}")
            return result_text, []

        except Exception as e:
            error_message = str(e)
            result_text = """
😔非常抱歉，您的查詢內容觸發了我們的內容過濾系統。為了確保對話的安全性和適當性，我無法直接回應您的問題。請您重新表述您的問題，避免使用可能引起敏感內容警報的字詞。如果您有任何疑問或需要協助，請隨時告訴我，我會盡力為您提供適當的幫助。謝謝您的理解與配合。
"""
            print(f"Unexpected error in chat_with_follow_up: {error_message}")
            return result_text, []
    
    def format_conversation_history(self, conversation_id: str) -> str:
        """Format conversation history for context"""
        if conversation_id not in self.conversations_memory:
            return ""
        
        history = self.conversations_memory[conversation_id]
        return "\n".join(f"Human: {turn['human']}\nAI: {turn['ai']}" for turn in history[-5:])




    



