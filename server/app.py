import os
import uuid
import hashlib
import openai
from flask import Flask,session
from flask import render_template
from flask import request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

app = Flask(__name__)
app.secret_key = 'chatdataqa'
app.config['UPLOAD_FOLDER'] = 'upload/'

openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ['OPENAI_API_BASE']
llm_name = "gpt-3.5-turbo"

persist_directory = 'docs/chroma/'

class QAChain:

    QA_CHAIN_PROMPT = None  
    qa_chain = None

    def __init__(self) -> None:       
         # Build prompt
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        self.QA_CHAIN_PROMPT = PromptTemplate.from_template(template)       

    def load_db(self, filepath):
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        # split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        # define embedding
        embeddings = OpenAIEmbeddings()
        # create vector database from data
        db = Chroma.from_documents(persist_directory=persist_directory,
                                documents=docs, embedding=embeddings)
        # define retriever
        retriever = db.as_retriever(search_type="similarity")
            
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer")
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=llm_name, temperature=0),
            chain_type="stuff",
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": self.QA_CHAIN_PROMPT},
            memory=memory,
            return_source_documents=True
        )       

qa = QAChain()

@app.route('/')
def index():
    return render_template('index.html')

basedir = os.path.abspath(os.path.dirname(__file__))

@app.route('/upload', methods=['POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        if f is None or f.filename == '':
            errorMsg = '请选择文件'
        elif f.mimetype != "application/pdf":
            errorMsg = '暂时只支持PDF文件'
        else:
            file_extension = os.path.splitext(f.filename)[1]
            folder_path = os.path.join(basedir,app.config['UPLOAD_FOLDER'])
            if not os.path.exists(folder_path):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)
            filepath = folder_path +create_uuid_from_string(f.filename) + file_extension
            app.logger.info(f"保存文件路径：{filepath}")                    
            f.save(filepath)
            qa.load_db(filepath)
            errorMsg = 'file uploaded successfully'                   
            f.close()
    else:
        errorMsg = 'Method not allowed'
    return {"errorMsg":errorMsg
            }


def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    search = data['search']    
    app.logger.info(f"问题：{search}")
    qa_chain = qa.qa_chain
    if qa_chain is None:
        app.logger.warn("未加载数据")
        return {
            "answer": "未加载数据"
        }
        
    result = qa_chain({"question": search})
    answer = result['answer']
    app.logger.info(f"答案：{answer}")

    return {
        "answer": answer
    }

if __name__ == '__main__':
    app.run()
