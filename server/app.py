import os,sys
import uuid
import hashlib
import openai
import langchain
import logging
from uuid import UUID
from typing import Any, Optional
from tenacity import RetryCallState
from flask import Flask
from flask import render_template, redirect, url_for
from flask import request,session, Response, stream_with_context
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.cache import InMemoryCache
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult

app = Flask(__name__)
app.secret_key = 'chatdataqa'
app.config['UPLOAD_FOLDER'] = 'upload/'

formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(thread)d --- [%(filename)s:%(lineno)d] - %(message)s")
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)
app.logger.handlers[0] = stream_handler

openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ['OPENAI_API_BASE']
llm_name = "gpt-3.5-turbo"

langchain.llm_cache = InMemoryCache()

persist_directory = 'docs/chroma/'

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.tokens = []
        # 记得结束后这里置true
        self.finish = False

    def on_llm_new_token(self, token: str, **kwargs):     
        self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.finish = True

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        app.logger.error(error)
        self.tokens.append(error.user_message)

    def generate_tokens(self):
        while not self.finish or self.tokens:
            if self.tokens:
                data = self.tokens.pop(0)
                yield data
            else:
                pass

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on a retry event."""
        app.logger.info("忽略重试")

class QAChain:
    chainStreamHandler = ChainStreamHandler()
    # 存储对话关联的qa chain
    conversationChain = dict()
    FILEPATH_KEY_PREFIX = "filepath:"

    def __init__(self) -> None:               
        # Build prompt
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        self.__QA_CHAIN_PROMPT = PromptTemplate.from_template(template)       

    def load_db(self, filepath, conversation_id):      
        session[QAChain.FILEPATH_KEY_PREFIX + conversation_id] = filepath
       
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
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        # 使用更便宜、更快的模型来完成问题的凝练工作，然后再使用昂贵的模型来回答问题
        QAChain.conversationChain[conversation_id] = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(streaming=True, max_tokens=4097, callbacks=[QAChain.chainStreamHandler], model_name=llm_name, temperature=0),
            chain_type="stuff",
            condense_question_llm = ChatOpenAI(temperature=0),
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": self.__QA_CHAIN_PROMPT},
            memory=memory,            
            return_source_documents=True
        )  
             

@app.route('/')
def index():
    chat_id = ""
    try:
        key = list(session.keys())[-1]      
        chat_id = key.split(":")[1]
        app.logger.debug(f"chat_id: {chat_id}")
    except Exception:
        chat_id = str(uuid.uuid1())
    
    return render_template('index.html', chat_id=chat_id)

basedir = os.path.abspath(os.path.dirname(__file__))

@app.route('/upload', methods=['POST'])
def uploader():
    if request.method == 'POST':
        conversation_id = request.form['conversation_id']
        if '-' not in conversation_id:
            return redirect(url_for('._index'))
        
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
            app.logger.debug(f"保存文件路径：{filepath}")                    
            f.save(filepath)
            qa = QAChain()
            qa.load_db(filepath, conversation_id)
            errorMsg = 'file uploaded successfully'                   
            f.close()
    else:
        errorMsg = 'Method not allowed'
    return {
        "errorMsg":errorMsg
    }


def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))

@app.route('/query', methods=['POST'])
def query():
    conversation_id = request.json['conversation_id']
    if '-' not in conversation_id:
        return redirect(url_for('._index'))
    
    data = request.get_json()
    search = data['search']        
    app.logger.debug(f"问题：{search}")
   
    qa_chain = QAChain.conversationChain.get(conversation_id)
    if qa_chain is None:
        filepath = session.get(QAChain.FILEPATH_KEY_PREFIX + conversation_id)
        if filepath is not None: 
            qa = QAChain()
            qa.load_db(filepath, conversation_id)            
            qa_chain = QAChain.conversationChain.get(conversation_id)
        else:
            return {
                "answer": "未加载数据"
            }        

    qa_run(qa_chain, search)
    return Response(stream_with_context(QAChain.chainStreamHandler.generate_tokens()), mimetype="text/event-stream")   

def qa_run(qa, question):
    qa({"question": question})

if __name__ == '__main__':
    app.run()
