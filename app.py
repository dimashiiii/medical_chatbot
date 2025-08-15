import streamlit as st
from helper import emdeddings  # Assuming this is correctly spelled as 'embeddings' in your helper file
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from prompt import *
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Set up the vector store and retriever
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=emdeddings  # Assuming this is your embedding function
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Set up the chat model
chatModel = ChatOpenAI(model="gpt-4o")

# Contextualize question prompt (for history-aware retrieval)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    chatModel, retriever, contextualize_q_prompt
)

# Update the main prompt to include chat history
# Assuming your original system_prompt includes {context} for the retrieved documents,
# e.g., "Answer the question based on the following context: {context}"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the chains
question_answer_chain = create_stuff_documents_chain(chatModel, prompt=prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Streamlit app
st.title("Medical Chatbot with Memory")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask a medical question..."):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Build chat_history from previous messages
    chat_history = []
    for message in st.session_state.messages[:-1]:  # Exclude the current user input
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        else:
            chat_history.append(AIMessage(content=message["content"]))
    
    # Generate response with chat history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            answer = response["answer"]
            st.markdown(answer)
    
    # Append assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})