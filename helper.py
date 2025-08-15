from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.embeddings import OpenAIEmbeddings

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_documents(documents: List[Document]) -> List[Document]:
    minimal_docs : List[Document] = []
    for doc in documents:
       src = doc.metadata.get("source")
       minimal_docs.append(
           Document(
               page_content=doc.page_content,
               metadata={"source": src}
           )
       )
    return minimal_docs

def text_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

emdeddings = OpenAIEmbeddings()
