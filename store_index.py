from dotenv import load_dotenv
import os 
from helper import load_pdf_files, filter_documents, text_splitter, emdeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_files("data/")
filtered_docs = filter_documents(extracted_data)
split_docs = text_splitter(filtered_docs)
emdeddings = emdeddings

print('divided documents into chunks')

pinecone = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"


if not pinecone.has_index(index_name):

    pinecone.create_index(
        index_name,
        dimension=1536,
        metric="cosine",
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pinecone.Index(index_name)

# if you want to create a new index and add documents to it, uncomment the following lines:
# docsearch = PineconeVectorStore(
#    embedding=emdeddings,    
#    index_name=index_name,
# )
# for i in range(0, len(split_docs), 100):
#     batch = split_docs[i:i+100]
#     docsearch.add_documents(batch)
#     print(f"Added {len(batch)} documents to the index.")


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=emdeddings
)

print(f"Connected to index: {index_name}")



