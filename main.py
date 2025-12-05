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

from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
store = {}
def get_session_history(session_id: str):
  if session_id not in store:
    store[session_id]=ChatMessageHistory()
  return store[session_id]



  

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", system_prompt + "\n\n{chat_history}"),
    ("human", "{input}"),
  ]
)
QA_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, QA_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# jinja templates
templates = Jinja2Templates(directory="template")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
  return templates.TemplateResponse("chatbot.html", {"request": request})

@app.post("/get")
def get_response(msg: str = Form(...), session_id: str = Form(...)):
  
  response = conversational_rag_chain.invoke({"input": msg},config={"configurable": {"session_id": session_id}})
  return str(response["answer"])

@app.post("/clear")
def clear_chat(session_id: str = Form(...)):
    store[session_id] = ChatMessageHistory()
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

  