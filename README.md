# Medical Chatbot with Memory

A RAG (Retrieval-Augmented Generation) powered medical chatbot built with Streamlit that maintains conversation history and provides contextually aware responses.

## Technical Overview

This application combines:
- **RAG Architecture**: Retrieves relevant medical information from a vector database before generating responses
- **Conversation Memory**: Maintains chat history for contextually aware follow-up questions
- **Vector Search**: Uses Pinecone for semantic similarity search across medical documents
- **LLM Integration**: GPT-4o processes queries and generates human-like responses

## Features

- 🧠 **Memory-enabled conversations** - Remembers previous interactions
- 🔍 **Semantic search** - Finds relevant medical information from knowledge base
- 💬 **Interactive UI** - Clean Streamlit interface
- 🎯 **Context-aware responses** - Uses chat history to better understand follow-up questions

## Architecture

```
User Query → History-Aware Retriever → Vector Search (Pinecone) → Document Retrieval → LLM (GPT-4o) → Response
```

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key
- Pre-populated Pinecone index named `medical-chatbot`

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install streamlit langchain-pinecone langchain-openai python-dotenv
```

3. Create `.env` file:
```env
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

4. Ensure you have:
   - `helper.py` with embeddings function
   - `prompt.py` with system_prompt variable

### Running the App

```bash
streamlit run app.py
```

## Key Components

- **Vector Store**: Pinecone index for medical document embeddings
- **Retriever**: Finds top 5 most relevant documents per query
- **History-Aware Retrieval**: Reformulates questions considering chat context
- **RAG Chain**: Combines retrieval and generation for informed responses

## Usage

1. Open the Streamlit interface
2. Ask medical questions in natural language
3. The bot will search relevant documents and provide informed responses
4. Follow-up questions automatically consider previous conversation context

## Note

This is for educational/research purposes. Always consult healthcare professionals for medical advice.
