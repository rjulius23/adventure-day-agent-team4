import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    VectorizedQuery
)

app = FastAPI()

load_dotenv()

class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    true_or_false = "true_or_false"
    popular_choice = "popular_choice"
    estimation = "estimation"

class Ask(BaseModel):
    question: str | None = None
    type: QuestionType
    correlationToken: str | None = None

class Answer(BaseModel):
    answer: str
    correlationToken: str | None = None
    promptTokensUsed: int | None = None
    completionTokensUsed: int | None = None

client: AzureOpenAI

if "AZURE_OPENAI_API_KEY" in os.environ:
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
index_name = "movies-semantic-index"
service_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_COMPLETION_MODEL")
credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"]) if len(os.environ["AZURE_AI_SEARCH_KEY"]) > 0 else DefaultAzureCredential()

# # Redis connection details
# redis_host = os.getenv('REDIS_HOST')
# redis_port = os.getenv('REDIS_PORT')
# redis_password = os.getenv('REDIS_PASSWORD')
 
# # Connect to the Redis server
# conn = redis.Redis(host=redis_host, port=redis_port, password=redis_password, encoding='utf-8', decode_responses=True)
 
# if conn.ping():
#     print("Connected to Redis")

@app.get("/")
async def root():
    return {"message": "Hello Smorgs"}

@app.post("/ask", summary="Ask a question", operation_id="ask") 
async def ask_question(ask: Ask):
    """
    Ask a question
    """

    start_phrase =  ask.question
    question_type = ask.type
    response: openai.types.chat.chat_completion.ChatCompletion = None

    if question_type == QuestionType.multiple_choice:
        sys_prompt = "Answer the question of the user and use the tools available to you. Only return the answer without encapsulating it in a sentence to the question what the tool returned in the content field, just the option not the index of the choice, no bs."
    elif question_type == QuestionType.true_or_false:
        sys_prompt = "Answer the question of the user and use the tools available to you. Only return the True or False without any making it a sentence it based on what the tool returned in the content field, no bs."
    elif question_type == QuestionType.estimation:
        sys_prompt = "Answer the question of the user and use the tools available to you. Only return the answer without encapsulating it in a sentence to the question what the tool returned in the content field, no bs."
    else:
        sys_prompt = "Answer the question of the user and use the tools available to you. Only return the answer without encapsulating it in a sentence to the question what the tool returned in the content field, no bs."

    index_name = "question-semantic-index"
    print(start_phrase)

    # create new searchclient using our new index for the questions
    search_client = SearchClient(
        endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"], 
        index_name=index_name,
        credential=credential
    )

    
    # check if the question &  answer is in the cache already
    vector = VectorizedQuery(vector=get_embedding(ask.question), k_nearest_neighbors=5, fields="vector")
    found_questions = list(search_client.search(
        search_text=None,
    query_type="semantic",
    semantic_configuration_name="question-semantic-config",
    vector_queries=[vector],
    select=["question", "answer"],
    top=5
    ))
    questionMatchCount = len(found_questions) 
    print(found_questions) 
    
    #  put the new question & answer in the cache as wel
    docIdCount = search_client.get_document_count()  +1 

    if(questionMatchCount>0):
        print ("Found a match in the cache.")
        # put the new question & answer in the cache as well
        #search_client.upload_documents(found_questions[0])
        best_match = max(found_questions, key=lambda x: x.get('@search.score', 0))
        print(best_match)
        # return the answer
        if best_match["@search.score"] > 0.95:
            search_client.upload_documents({'question': ask.question, 'answer': best_match["answer"], 'id': str(docIdCount), 'vector': get_embedding(ask.question)})
            return Answer(answer=best_match["answer"], correlationToken=ask.correlationToken, promptTokensUsed=0, completionTokensUsed=0)
    
    print("No match found in the cache.")        
        
    #   reach out to the llm to get the answer. 
    print('Sending a request to LLM')
    start_phrase = ask.question
    messages=  [{"role" : "assistant", "content" : start_phrase},
                    { "role" : "system", "content" : sys_prompt}]
        
    response = client.chat.completions.create(
            model = deployment_name,
            messages =messages,
    )
    answer = Answer(answer=response.choices[0].message.content)

    #  put the new question & answer in the cache as wel
    docIdCount = search_client.get_document_count()  +1 
        
    search_client.upload_documents({'question': ask.question, 'answer': answer.answer, 'id': str(docIdCount), 'vector': get_embedding(ask.question)})


    print ("Added a new answer and question to the cache: " + answer.answer + "in position" + str(docIdCount))
    
    answer = Answer(answer=response.choices[0].message.content)
    answer.correlationToken = ask.correlationToken
    answer.promptTokensUsed = response.usage.prompt_tokens
    answer.completionTokensUsed = response.usage.completion_tokens

    return answer

# use an embeddingsmodel to create embeddings
def get_embedding(text, model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")):
    return client.embeddings.create(input = [text], model=model).data[0].embedding