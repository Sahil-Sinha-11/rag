from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

path_pdf = Path(__file__).parent / "node.pdf"

loader = PyPDFLoader(file_path=path_pdf)
docs=loader.load()

# print (docs[12])

# splitting docs in samller chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents=docs)

#vectrize the chunks

# vector embeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store=QdrantVectorStore.from_documents(
    documents = chunks,
    embedding=embedding_model,
    url = "http://localhost:6333",
    collection_name="langchain-docs"
)

print("indexing done");