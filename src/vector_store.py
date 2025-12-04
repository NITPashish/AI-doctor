from dotenv import load_dotenv
import os
from src.helper import load_doc, filtered_doc, text_split, load_embeddings
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# load pdf
extracted_data = load_doc(data="book/")
filtered_data = filtered_doc(extracted_data)
text_chunks = text_split(filtered_data)

embeddings = load_embeddings()


from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

from pinecone import ServerlessSpec 

index_name = "ai-doctor"
if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  
        metric= "cosine",  
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore


vectorData = PineconeVectorStore.from_documents(
  documents=text_chunks,
  embedding=embeddings,
  index_name=index_name
)

# use it from existing index
# vectorData = PineconeVectorStore.from_existing_index(
#   embedding=embeddings,
#   index_name=index_name
# )

# retreival
retriever = vectorData.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})
