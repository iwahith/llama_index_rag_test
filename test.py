from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import os
import time
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine
from dotenv import load_dotenv
from typing import List
import json
import glob
from llama_parse import LlamaParse
from llama_index.core import StorageContext
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAgent
from pydantic import BaseModel


nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = "sk-"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)






llm = OpenAI(model="gpt-4-1106-preview")
text_splitter = SentenceSplitter(chunk_size=6000, chunk_overlap=200)
embed_model = OpenAIEmbedding()

Settings.llm = llm
Settings.text_splitter = text_splitter
Settings.embed_model = embed_model

response_synthesizer = get_response_synthesizer(response_mode="compact")

parser = LlamaParse(
    api_key="llx-",
    result_type="markdown",
    verbose=True,
)

file_extractor = {".pdf": parser}




class document_info(BaseModel):
    """Title, Summary and Question from the document"""
    
    Title: str
    Summary: str
    Question: List[str]

def save_uploaded_file(file: UploadFile, upload_folder: str):
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path

agent = None
file_info_list = []
sub_tools = []
fin_tools = []

@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    try:
        
        filenames = [file.filename for file in files]
        f = ", ".join(filenames) 
        print(f)
        
        filepath_with_underscores = [filename[:-4].replace(" ", "_").replace("-", "_") for filename in filenames]
        doc_str = ", ".join(filepath_with_underscores)
        cleaned_names = [file_name.replace('.pdf', '').replace(' ', '_') for file_name in filenames]
        output_cleaned_names = '_'.join(cleaned_names)
        print("File Uploading")
        global agent
        for i, file in enumerate(files):
            file_path = save_uploaded_file(file, UPLOAD_FOLDER)
            file_info_list.append({"file_name": file.filename, "file_path": file_path})
            documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
            index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter]) # , transformations=[text_splitter]
            
            summary_query_engine = index.as_query_engine(response_mode="tree_summarize", output_cls=document_info)
            output = json.loads(str(summary_query_engine.query("Extract the title, concise 2-3 line summary and Generate 5 questions of the document")))

            vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
            vector_query_engine = RetrieverQueryEngine(retriever=vector_retriever, response_synthesizer=response_synthesizer)
            


            print("Chain Created")
            qa_tool = QueryEngineTool(
        query_engine=vector_query_engine, 
        metadata=ToolMetadata(
            name=filepath_with_underscores[i],
            description="Provides information about {file} document."
            "Use a detailed plain text question as input to the tool."
            )
        )
            sub_tools.append(qa_tool)
            
            
            print("Tools Created")
        
        agent = OpenAIAgent.from_tools(
        sub_tools,
        llm=llm,
        verbose=True,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about the """+doc_str+""" document.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )
    



        print("Created Knowledge Based, Chain, Tools & Agents")
        file_paths = [file_info['file_path'] for file_info in file_info_list]
        print("File Uploaded")
        return {"file_paths": file_paths, "title_info":[output] ,"generated_questions": output['Question']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_response(message: str) -> str:
    response = agent.chat(message)
    return response.response

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, client_id: int):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: int):
        del self.active_connections[client_id]
        global sub_tools, fin_tools
        sub_tools = []
        fin_tools = []

    async def send_personal_message(self, message: str, client_id: int):
        await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            response = generate_response(data)
            await manager.send_personal_message(response, client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        pdf_docs = glob.glob('./uploads/*')
        for f in pdf_docs:
            os.remove(f)







