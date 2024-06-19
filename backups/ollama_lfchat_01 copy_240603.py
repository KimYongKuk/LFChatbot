import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
# import ast
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import streamlit.components.v1 as components
from ollama import Client
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import torch
import uuid
import chromadb
from chromadb.utils import embedding_functions

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# from langchain_chroma import chroma
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)

from tqdm import tqdm

load_dotenv()

st.set_page_config(page_title='L&F Chat ', page_icon=':🐳:', layout="wide")

## title    



st.error('🚨 검색어에 주의하세요, 불필요한 검색은 삼가해주세요!')
colored_title = """
    <h1 style="margin-top:20px;">
    <span style="color: white;">WELCOME TO</span><span style="color:#FF3F00;">&nbsp; L&F BOT🔥</span>
</h1>
"""
if "colored_title" not in st.session_state:
  st.session_state["colored_title"] = colored_title

st.markdown(st.session_state["colored_title"], unsafe_allow_html=True)

with st.sidebar:
    # with st.echo():
    #     st.write("사이드바")

    # with st.spinner("Loading..."):
    #     time.sleep(5)
    st.success('🔥 주의사항에 대한 안내를 확인하세요. http://lfon.landf.co.kr')
    st.markdown('---')
    # st.success('🔥 아래는 검색어 예시입니다.')
    st.write('<p style="font-size: 17px;"><span class="bold">🤗아래는 검색어 예시입니다🤗</span></p> ', unsafe_allow_html=True)

    st.markdown('<p style="font-size: 15px;"> # MES 담당자가 누구야? </p>\
                 <p style="font-size: 15px;"> # LFON 계정을 생성하고 싶어. </p>\
                 <p style="font-size: 15px;"> # 포맷 하고싶은데 어떻게해야해? </p>\
                 <p style="font-size: 15px;"> # L&F의 조직문화에 대해 알려줘. </p>\
                 <p style="font-size: 15px;"> # 전표 프로세스가 궁금한데 알려줄래? </p>', unsafe_allow_html=True)



#CONST
################################
NAME = '김용국(wg0403)' 
CHROMADB_PATH   = "./chdata" 
CHUNK_SIZE = 4096
res_message = 'test'
################################

#PersistentClient의 경우, 영구적으로 디스크에 저장 가능함
#VectorDB Chroma의 client 생성.
perClient = chromadb.PersistentClient(
    path=CHROMADB_PATH
)

#Collection 생성.
posts = perClient.get_or_create_collection (
    name="lftest01",
    # metadata={'hnsw:space': 'cosine'},
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
) #bespin-global/klue-sroberta-base-continue-learning-by-mnr

#임베딩 모델, dimension 768
embeddingModel = SentenceTransformerEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")

#set vectorStore
vectorStore = Chroma(
    client = perClient,
    collection_name="lftest01",
    embedding_function=embeddingModel
)

#return embedding model
def test_embedding(text):
    text = text.replace("\n", " ")
    model = SentenceTransformer("bespin-global/klue-sroberta-base-continue-learning-by-mnr")
    embeddings = model.encode(text)
    return embeddings

def prompt_clear():
    st.session_state["generated"] = []
    st.session_state['past'] = []

def get_embedding(text, engine): # 입력받은 소스 텍스트와, 임베딩 모델을 통해 임베딩을 진행하는 함수.
    text = text.replace("\n", " ")
    return clientEmbedding.embeddings.create(input = [text], model=engine).data[0].embedding

# 두 임베딩간 유사도 계산
def cos_sim(A, B):
    return dot(A, B)  / (norm(A)*norm(B))

# 질문을 임베딩하고, 유사도 높은 탑3 자료
def return_answer_candidate(df, query):
    query_embedding = get_embedding(
        query,  
        engine="text-embedding-ada-002"
    )

    #model import
    model = SentenceTransformer("bespin-global/klue-sroberta-base-continue-learning-by-mnr")
    # sentences = ["This is an example sentence", "Each sentence is converted"]
    query_embedding2 = model.encode(query)

    print('###1번###')
    print(query_embedding)
    print('###2번###')
    print(query_embedding2)

    print('###3번###')
    query_embedding2_float = [tensor.item() for tensor in query_embedding2]
    print(query_embedding2_float)

    # 입력된 질문과 각 문서의 유사도
    df['similarity'] = df['embedding'].apply(lambda x: cos_sim(np.array(query_embedding2_float), np.array(x)))
    # 유사도 높은 순으로 정렬
    top2 = df.sort_values("similarity", ascending=False).head(3)
    return top2

#split text
def split_Value():
    loader = DirectoryLoader('./data', glob="*.txt", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents(documents)
    return texts

#return answer in vectorDB
def answer_vectorDB(query): # query = "경영정보팀에 대해 알려줘"
    docs = vectorStore.similarity_search(query)
    return docs
    # print('정답1★' + docs[0].page_content)
    # print('#############################')
    # print(vectorStore._collection.count(), "in the collection")


# text파일을 임베딩 csv로 변환하는 함수
def create_embedding(txt_file_name, folder_path):
 #split 문서정보를 Collection에 add.
    tmpIdx = 0
    for doc in split_Value():
        tmpIdx += 1
        posts.add(
            ids=['문서' + str(tmpIdx)], documents=doc.page_content
        )

    #lftest01 collection으로 vectorstore 생성
  

    # vectorStore = Chroma.from_documents(
    #     documents = split_Value(),
    #     embedding = embeddingModel,
    #     persist_directory="./data"
    # )

    # perClient.delete_collection("lftest01")
    # print(posts.get())

# 질문에 대한 가장 유사한 문서3개 가져와서, messages셋 만들어서 리턴    
def create_prompt(query):
    
    # 질문과 가장 유사한 문서 3개 가져오기
    result = answer_vectorDB(query)
    print('###데이터프레임###')
    print(result) # 결과 DataFrame 출력

    
    # 문서 내용을 동적으로 포맷팅하기 위한 코드
    system_message_parts = []
    system_message_parts.extend(doc.page_content for doc in result)

    # 모든 문서 부분을 개행 문자로 구분하여 하나의 문자열로 합침
    documents_formatted = "\n".join(system_message_parts) + "\n"

    print('###문서 포맷###')
    print(documents_formatted)

    # 최종 system_message에 문서 내용 포함
    system_message = f"""
    나는 L&F봇입니다. 나는 L&F의 사내정책, 문화, 조직에 대해 알려주기 위해 만들어진 챗봇입니다.
    아래 세 개의 역따옴표로 구분된 텍스트는 사내 문서입니다.
    ```{documents_formatted}```
    1. 사내 문서에 근거해서 user content 질의에 대해서 최대한 간결하게 답하세요.
    2. user content 질의내용이 사내 문서에 없다면, '죄송합니다. 그 부분은 모르겠습니다.' 라고만 답해야 합니다.
    3. 링크가 있다면 답변에 꼭 포함하여 주세요.
    4. 무조건 한글로 대답하세요.
    """

    user_message = f"""User question: "{str(query)}". """

    messages =[
        {"role": "user", "content": user_message},
        {"role": "system", "content": system_message}
    ]
    # print(messages)
    return messages


# 완성된 질문에 대한 답변 생성
def generate_response(messageArr):
    res_message = []

    client = Client(host='http://192.168.120.253:11434')

    # response = client.chat(model='llama-3-alpha-ko-8b-instruct-q4_k_m', messages=messageArr,
    response = client.chat(model='gemma', messages=messageArr,
        stream = True,
        options = {
            # "temperature": 0,
            # "top_k":10,
            # "top_p":0.5,
            # "num_ctx":500,
            # "mirostat_tau":1.0,
            # "mirostat":1
            # "stop" : ["\n", "----------------------------------------------------"]
        }
    )
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)
        token = chunk['message']['content']
        res_message += token
        yield chunk['message']['content']
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "https://media.bunjang.co.kr/product/254293734_1_1708650713_w360.jpg" if message["role"] == "user" else "🐌"

    with st.chat_message(message["role"], avatar = avatar):
        st.markdown(message["content"])

#############################################################################
#############################################################################

folder_path = './data'  
# 폴더 및 모든 파일 중 txt파일만 리스트로 가져와서 for 문 시작
txt_file_name = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

# test_embedding('test')
#### main logic
create_embedding(txt_file_name, folder_path)

# React to user input
if user_input := st.chat_input("'LFON 계정생성을 하고싶어', '경영정보팀에 대해 알려줘'"):
    # Display user message in chat message container
    # st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="https://media.bunjang.co.kr/product/254293734_1_1708650713_w360.jpg"):
        st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
    with st.chat_message("assistant", avatar="🐌"):
        prompt = create_prompt(user_input)
        response = st.write_stream(generate_response(prompt))
        print('###사용자 질의###')
        print(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})