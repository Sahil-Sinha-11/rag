from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

openai_client = OpenAI()


# vector embeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url = "http://localhost:6333",
    collection_name="langchain-docs"
)


# taking user querry
user_querry = input("Ask Something: ")

# Relevent chunks from vector db

search_results = vector_db.similarity_search(query=user_querry)

context = "\n\n\n".join([
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata['page_label']}\n"
    f"File Location: {result.metadata['source']}"
    for result in search_results
])

SYSTEM_PROMPT = f"""
 You are a helpful AI Assistant who answers querry based on available context
 retrived from a pdf file along with page_contents and numbers. If you don't know the answer, say you don't know. 
    Context: {context}
"""


response = openai_client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_querry},
    ]
)

print(f"🤖 : Response: {response.choices[0].message.content}")