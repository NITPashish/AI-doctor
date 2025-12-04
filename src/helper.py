from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_doc(data):
  loader = DirectoryLoader(
    data,
    glob="*.pdf",
    loader_cls=PyPDFLoader
  )
  documents = loader.load()
  return documents


from typing import List
from langchain.schema import Document

def filtered_doc(loaded_doc: List[Document]) -> List[Document] :
  updated_doc : List[Document] = []
  for doc in loaded_doc:
    src = doc.metadata.get("source")
    page = doc.metadata.get("page")
    updated_doc.append(
      Document(
        page_content=doc.page_content,
        metadata = {"source": src, "page": page}

      )
    )
  return updated_doc

# split the docs into smaller chunks
def text_split(docs):
  text_split = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 20
  )
  text = text_split.split_documents(docs)
  return text


from langchain.embeddings import HuggingFaceEmbeddings

def load_embeddings():
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  embeddings = HuggingFaceEmbeddings(model_name=model_name)
  return embeddings