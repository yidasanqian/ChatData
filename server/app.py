import os
import openai
from flask import Flask
from flask import render_template
from flask import request
from dotenv import load_dotenv, find_dotenv
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

#load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ['OPENAI_API_BASE']
llm_name = "gpt-3.5-turbo"



@app.route('/')
def index():
    return render_template('index.html')

# load
persist_directory = 'docs/chroma/'
file = "docs/Java开发手册（嵩山版）.pdf"
def load_db():
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = Chroma.from_documents(persist_directory=persist_directory,
                               documents=docs, embedding=embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity")
    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer")
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type="stuff", 
        retriever=retriever, 
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        memory=memory,
        return_source_documents=True
    )
    return qa 

qa_chain = load_db()

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    search = data['search']
    app.logger.info(f"问题：{search}" )
    result = qa_chain({"question":search})
    answer = result['answer']
    app.logger.info(f"答案：{answer}" )

    return {
        "answer": answer
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)