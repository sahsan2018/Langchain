from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from embeddings import chat_model, embeddings
from langchain.chains import create_history_aware_retriever



system_prompt = (
    "find the courses that match the student query and answer according to the question. if a student ask you to list all courses till eighth semester then list all the courses till 8th semester from the given datatset. Do not miss any class for god sake."
                    "Consider the prerequisites and corequisites for each course when making recommendations."
                    "Pre-requisite: Classes you must take prior to the specific class."
                    "Co-requisite: Classes you can take along with the desired class if you have not taken them before."
                    "For example, suppose someone took CET1111, CET1150, ENG1101 so far."
                    "As they have not completed all classes needed for starting second-semester classes,"
                    "they cannot take any classes which requires classes other than the completed ones, so first, they can take CET 1120, CET1100 and some other classes which does not require prerequisite of other than the completed ones."
                    "Have a user-friendly but straightforward conversation and do not use unnecessary sentences.Provide a detailed and accurate list of all courses a student needs to take from the first semester "
                    "to the eighth semester to graduate. Include each course's name, prerequisites, corequisites, and semester. "
                    "Base your answer on the program's curriculum guidelines and ensure no course is omitted. If a student asks for "
                    "graduation requirements, confirm you cover all required general education and major-specific courses"
    "{context}"
)

import os
from pathlib import Path

# Get the current file's directory
current_dir = Path(__file__).parent

vector = FAISS.load_local(r"faissindexupdate62", embeddings, allow_dangerous_deserialization=True)

# Create retriever and retrieval chain
retriever = vector.as_retriever()
# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    chat_model, retriever, 
    ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question..."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
)

# Create final question-answering chain
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)

# Create RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



