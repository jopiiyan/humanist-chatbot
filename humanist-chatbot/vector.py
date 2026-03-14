from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

BASE_DIR = os.path.dirname(__file__)

PDF_FOLDER = os.path.join(BASE_DIR, "articles")

# Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Vector DB location
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

documents = []
ids = []

if add_documents:

    # Load each PDF
    for filename in os.listdir(PDF_FOLDER):

        if filename.endswith(".pdf"):

            filepath = os.path.join(PDF_FOLDER, filename)

            loader = PyPDFLoader(filepath)

            pdf_docs = loader.load()

            documents.extend(pdf_docs)

    # Split large text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(documents)

    # Assign IDs
    ids = [str(i) for i in range(len(split_docs))]

else:
    split_docs = []

# Create vector store
vector_store = Chroma(
    collection_name="humanist_articles",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents only first time
if add_documents:

    vector_store.add_documents(
        documents=split_docs,
        ids=ids
    )

# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)