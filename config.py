import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDaUzWnAuFZDx1O6J_Xc_AEcZwkiW0aOOE")
