from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import getpass
import os

from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize embeddings and chat model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,

)

