import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

app = FastAPI()

load_dotenv()

class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    true_false = "true_or_false"
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

@app.get("/")
async def root():
    return {"message": "Hello Smoorghs"}

@app.post("/ask", summary="Ask a question", operation_id="ask") 
async def ask_question(ask: Ask):
    """
    Ask a question
    """

    # Send a completion call to generate an answer
    print('Sending a request to openai')
    start_phrase =  ask.question
    question_type = ask.type
    response: openai.types.chat.chat_completion.ChatCompletion = None

    # Send a completion call to generate an answer

    if question_type == QuestionType.multiple_choice:
        response = client.chat.completions.create(
            model = deployment_name,
            messages = [{"role" : "assistant", "content" : start_phrase}, 
                     { "role" : "system", "content" : "Answer with the correct option w/o index, no bs:"}]
        )
    elif question_type == QuestionType.true_or_false:
        response = client.chat.completions.create(
            model = deployment_name,
            messages = [{"role" : "assistant", "content" : start_phrase}, 
                     { "role" : "system", "content" : "Answer true or false, no bs"}]
        )
    elif question_type == QuestionType.estimation:
        response = client.chat.completions.create(
            model = deployment_name,
            messages = [{"role" : "assistant", "content" : start_phrase}, 
                     { "role" : "system", "content" : "Give me a number, no bs"}]
        )
    else:
        response = client.chat.completions.create(
            model = deployment_name,
            messages = [{"role" : "assistant", "content" : start_phrase}, 
                     { "role" : "system", "content" : "Answer this question:"}]
        )

    answer = Answer(answer=response.choices[0].message.content)
    answer.correlationToken = ask.correlationToken
    answer.promptTokensUsed = response.usage.prompt_tokens
    answer.completionTokensUsed = response.usage.completion_tokens

    return answer
