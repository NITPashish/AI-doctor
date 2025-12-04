from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.helper import load_embeddings
from src.prompt import system_prompt

from dotenv import load_dotenv
import os
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI()


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "ai-doctor"
embeddings = load_embeddings()
vectorData = PineconeVectorStore.from_existing_index(
  embedding=embeddings,
  index_name=index_name
)

retriever = vectorData.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", system_prompt),
    ("human", "{input}"),
  ]
)
QA_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, QA_chain)



# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# jinja templates
templates = Jinja2Templates(directory="template")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
  return templates.TemplateResponse("chatbot.html", {"request": request})

@app.post("/get")
def get_response(msg: str = Form(...)):
  response = rag_chain.invoke({"input": msg})
  return str(response["answer"])
  