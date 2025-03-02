from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Store for managing session-based histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]