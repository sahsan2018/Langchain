from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY

# Initialize embeddings and chat model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
import getpass
import os

os.environ["OPENAI_API_KEY"] = "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"

from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,

)
