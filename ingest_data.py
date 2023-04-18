from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ReadTheDocsLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

# Load Data

def scrapeAndSave(url="langchain.readthedocs.io", out_path="vectorstore.pkl"):
    loader = ReadTheDocsLoader(url)
    raw_documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(raw_documents)


    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)


    # Save vectorstore
    with open(out_path, "wb") as f:
        pickle.dump(vectorstore, f)
