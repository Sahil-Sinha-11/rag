from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

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

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
