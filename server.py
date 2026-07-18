import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Enable CORS so your frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the initialized database components
vector_store = None
retriever = None
rag_chain = None

class QueryRequest(BaseModel):
    api_key: str
    query: str
    pdf_path: str

@app.post("/api/initialize")
async def initialize_database(request: QueryRequest):
    global vector_store, retriever, rag_chain
    
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="The specified PDF file path was not found on the server.")
        
    try:
        os.environ["OPENAI_API_KEY"] = request.api_key
        
        # Parse the document
        loader = PyPDFLoader(request.pdf_path)
        docs = loader.load()
        
        # Split text chunks semantically
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        
        # Generate embeddings and store locally in memory
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(final_documents, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Rigid prompt enforcement matching your specification constraints
        system_prompt = (
            "You are an Islamic Scholar AI assistant. Answer the question using ONLY the provided context below. "
            "If the answer cannot be found, respond exactly with: 'I could not find a reliable answer in the provided reference books.'\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return {"status": "success", "message": "Knowledge base successfully vectorized."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_knowledge_base(request: QueryRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=400, detail="The database must be initialized before querying.")
    
    try:
        os.environ["OPENAI_API_KEY"] = request.api_key
        response = rag_chain.invoke({"input": request.query})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
