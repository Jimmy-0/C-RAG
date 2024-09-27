import os

import openai
from openai import AzureOpenAI
from openai import AsyncOpenAI

from langchain_openai import AzureChatOpenAI  
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings

from app.utils.utils import measure_time

class OpenaiConfig:
    token="bc0865881852423cbf3534b07401b39b"
    endpoint="https://cathay-chm-ai.openai.azure.com/"
    type="azure"
    version="2023-03-15-preview"


    @staticmethod
    @measure_time
    def initial_openai():
        # OpenAI API configuration
        openai.api_type = OpenaiConfig.type
        openai.base_url  = OpenaiConfig.endpoint
        openai.api_version = OpenaiConfig.version
        openai.api_key = OpenaiConfig.token
        return openai
    
    @staticmethod
    @measure_time
    def initial_azureopenai():
        return AzureOpenAI(
            api_key = OpenaiConfig.token,
            # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
            api_version = OpenaiConfig.version,
            # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
            azure_endpoint = OpenaiConfig.endpoint,
        )
    
    @staticmethod
    @measure_time
    def initail_azurechatai_gpt4o(temperature=0):
        os.environ["AZURE_OPENAI_ENDPOINT"] = OpenaiConfig.endpoint
        os.environ["AZURE_OPENAI_API_KEY"] = OpenaiConfig.token
        return AzureChatOpenAI(  
            azure_deployment="gpt4o",  
            api_version="2024-05-01-preview",  
            model="gpt-4o",  
            model_version="2024-05-13",
            temperature=temperature
        )
    
    @staticmethod
    @measure_time
    def initail_azureopenai_embeddings():
        return AzureOpenAIEmbeddings(
            deployment="embeddings",
            model="text-embedding-ada-002",
            azure_endpoint=OpenaiConfig.endpoint,
            openai_api_type="azure",
            openai_api_key=OpenaiConfig.token
        )
    
    @staticmethod
    @measure_time
    def initial_openai_client():
        client = AsyncOpenAI(
            # This is the default and can be omitted
            api_key = OpenaiConfig.token,
            base_url = OpenaiConfig.endpoint,
            timeout = 3000
        )
        return client