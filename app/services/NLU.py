import os
import json
import time
from typing import Dict, List, Set
from pydantic import BaseModel

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI

# Local imports
from config import OpenaiConfig
from app.utils.utils import EntityType, HRIntentCategory, get_Chinese_intent

# Pydantic models
class Entity(BaseModel):
    value: str
    types: List[EntityType]

class NLUOutput(BaseModel):
    intent: HRIntentCategory
    entities: List[Entity]
    keywords: List[str]

# Environment setup
os.environ["AZURE_OPENAI_ENDPOINT"] = OpenaiConfig.endpoint
os.environ["AZURE_OPENAI_API_KEY"] = OpenaiConfig.token

# LLM setup
llm = AzureChatOpenAI(
    azure_deployment="gpt4o",
    api_version="2024-05-01-preview",
    model="gpt-4o",
    model_version="2024-05-13",
    temperature=0
)

# Prompt template
nlu_prompt = ChatPromptTemplate.from_messages([
    ("system", """您是一位專精於人力資源查詢的 AI 助理。請從人力資源的角度分析以下輸入：

    輸入：{input}

    請提取實體和關鍵字，並判斷意圖。使用以下預定義的實體類型和意圖類別：

    實體類型：
    {entity_types}

    意圖類別：
    {intent_categories}

    注意某些實體可能同時屬於多個類型。如果實體不屬於這些類型，請使用"其他"。


    請按照以下 JSON 格式結構化回應：
    {{
        "intent": "IntentCategory",
        "entities": [
            {{
                "value": "entites",
                "types": ["entity_type1", "entity_type2"]
            }}
        ],
        "keywords": ["keyword1", "keyword2"],
        "position": "identified job role or null"
    }}

    請確保對類似的查詢保持一致的分類，並反映實體之間的關係。

    之前的對話摘要：
    {history}
    """),
])

# Chain setup
memory = ConversationBufferMemory(memory_key="history", input_key="input")
nlu_chain = (
    {"input": RunnablePassthrough(), 
     "entity_types": lambda _: ", ".join([f"{e.name}: {e.value}" for e in EntityType]),
     "intent_categories": lambda _: "\n".join([f"{i.name}: {i.value}" for i in HRIntentCategory]),
     "history": lambda _: memory.load_memory_variables({})["history"]}
    | nlu_prompt
    | llm
    | RunnablePassthrough()
)

sensitive_word_set: Set[str] = {"性騷擾"}
sensitive_word_map: Dict[str, str] = {"性騷擾": "越矩的肢體碰觸行為"}


def NLU_classification(input_text: str) -> NLUOutput:

    start_time = time.time()
    try:
        result = nlu_chain.invoke(input_text)
        
        if isinstance(result, AIMessage):
            result = result.content
        
        output_dict = json.loads(result)
        # print(f"=========NLU=========")
        for i,v in enumerate(output_dict['keywords']):
            # print("======output_dict['keywords']======")
            print(f"{v}")
            if v in sensitive_word_set:
                print("Find the sensative word!")
                output_dict['keywords'][i] = sensitive_word_map.get(v,v)

            
        for e in output_dict['entities']:
            if e['value'] in sensitive_word_set:
                output_dict['entities'] = sensitive_word_map.get(e['value'],e['value'])

        # print(f"=========NLU=========")
        intent_str = output_dict['intent'].upper()
        
        try:
            intent = HRIntentCategory[intent_str]
        except KeyError:
            intent = HRIntentCategory.OTHER


        entities = []
        for entity_dict in output_dict.get('entities', []):
            try:
                entity_types = []
                for t in entity_dict['types']:
                    try:
                        # print("========entity_dict['types']=======")
                        print(t)
                        # print("========entity_dict['types']=======")
                        entity_types.append(EntityType[t.upper()])
                    except KeyError:
                        entity_types.append(EntityType.OTHER)
                
                entity = Entity(
                    value=entity_dict['value'],
                    types=entity_types
                )
                entities.append(entity)
            except Exception as e:
                continue
        
        output = NLUOutput(
            intent=intent,
            entities=entities,
            keywords=output_dict.get('keywords', [])
        )

        memory.save_context({"input": input_text}, {"output": result})
        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds")

        return output
    except Exception as e:
        return NLUOutput(intent=HRIntentCategory.OTHER, entities=[], keywords=[])

