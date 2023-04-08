from dotenv import load_dotenv
load_dotenv()

import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PagedPDFSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

loader = DirectoryLoader('', glob="pdfs/*.pdf", loader_cls=PagedPDFSplitter)

pages = loader.load_and_split()

def ingest_docs():
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()