import os
import openai
from openai import OpenAI
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import time
import streamlit.components.v1 as components
import random
import asyncio
import logging

load_dotenv()

st.set_page_config(page_title='L&F Chat ', page_icon=':🐳:', layout="wide")


# Set OpenAI API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get('OPENAI_API_KEY'),
)

def prompt_clear():
    st.session_state["generated"] = []
    st.session_state['past'] = []

def get_embedding(text, engine): # 입력받은 소스 텍스트와, 임베딩 모델을 통해 임베딩을 진행하는 함수.
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=engine).data[0].embedding


# 두 임베딩간 유사도 계산
def cos_sim(A, B):
    return dot(A, B)  / (norm(A)*norm(B))

# 질문을 임베딩하고, 유사도 높은 탑3 자료
def return_answer_candidate(df, query):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    # 입력된 질문과 각 문서의 유사도
    df['similarity'] = df['embedding'].apply(lambda x: cos_sim(np.array(query_embedding), np.array(x)))
    # 유사도 높은 순으로 정렬
    top2 = df.sort_values("similarity", ascending=False).head(2)
    return top2




# 질문에 대한 가장 유사한 문서3개 가져와서, messages셋 만들어서 리턴
def create_prompt(query):

    file_list = os.listdir(folder_path)

    all_data = []

    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, encoding='utf-8')
            df['embedding'] = df['embedding'].apply(lambda x: [float(num) for num in x.strip('[]').split(',')])
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # 질문과 가장 유사한 문서 3개 가져오기
    result = return_answer_candidate(combined_df, query)
    # print(result) # 결과 DataFrame 출력

    # 문서 내용을 동적으로 포맷팅하기 위한 코드
    system_message_parts = [
        f"문서{i + 1}: {text}"
        for i, text in enumerate(result['text'])  # result DataFrame에서 'text' 열의 모든 항목을 순회
    ]


    
    # 모든 문서 부분을 개행 문자로 구분하여 하나의 문자열로 합침
    documents_formatted = "\n".join(system_message_parts)

    # 최종 system_message에 문서 내용 포함
    system_message = f"""
    너의 이름은 L&F봇이야. 너는 L&F의 사내정책, 문화, 조직에 대해 알려주기 위해 만들어진 챗봇이야.
    너는 주어진 문서만을 참고해서 대답해줘. 질문과 문서와 관련없는 내용은 절대 답변하지마.
    문서내용:
    {documents_formatted}
    문서에 나와있는 건, 친절히 알려주되, 문서에 나와있지 않은 건 모른다고 대답해줘.
    무조건 존댓말로 대답해줘.
    """

    user_message = f"""User question: "{str(query)}". """

    messages =[
        {"role": "user", "content": user_message},
        {"role": "system", "content": system_message}
    ]
    # print(messages)
    return messages

# 완성된 질문에 대한 답변 생성
def generate_response(messages):
    result = client.chat.completions.create(
            stream=True,
            model="gpt-3.5-turbo",
            messages= messages,
            temperature = 0.5,
            max_tokens = 1000
        )
    # print(result.choices[0].message.content)
    return result #result.choices[0].message.content

#############################################################################
#############################################################################

folder_path = './data'
# 폴더 및 모든 파일 중 txt파일만 리스트로 가져와서 for 문 시작
txt_file_name = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

## title

st.error('🚨 검색어에 주의하세요, 불필요한 검색은 삼가해주세요!')
colored_title = """
    <h1 style="margin-top:20px;">
    <span style="color: white;">WELCOME TO</span><span style="color:#FF3F00;">&nbsp; L&F🔥</span>
</h1>
"""
if "colored_title" not in st.session_state:
  st.session_state["colored_title"] = colored_title

st.markdown(st.session_state["colored_title"], unsafe_allow_html=True)

#### main logic
data = []

for file in txt_file_name:
    if os.path.isfile(os.path.join(folder_path, file + '.csv')):
        continue
    
    data = []
    txt_file_path = os.path.join(folder_path, file)
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        file_name = file + '.csv'
        file_path = os.path.join(folder_path, file_name)
        text = f.read()
        data.append(text)
        df = pd.DataFrame(data, columns=['text'])
        df['embedding'] = df.apply(lambda row: get_embedding(row.text, engine="text-embedding-ada-002"), axis=1) # 임베딩 데이터 저장
        df.to_csv(file_path, index=False, encoding='utf-8') # csv 생성

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "https://media.bunjang.co.kr/product/254293734_1_1708650713_w360.jpg" if message["role"] == "user" else "🐌"

    with st.chat_message(message["role"], avatar = avatar):
        st.markdown(message["content"])
        
    

# React to user input
if user_input := st.chat_input("'LFON 계정생성을 하고싶어', '경영정보팀에 대해 알려줘'"):
    # Display user message in chat message container

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="https://media.bunjang.co.kr/product/254293734_1_1708650713_w360.jpg"):
        #https://media.bunjang.co.kr/product/254293734_1_1708650713_w360.jpg
        #https://image.fmkorea.com/files/attach/new3/20230714/486616/1934106089/5966131270/9a6909c7b9bc74cd9c1418e330bb9a88.jpeg
        #https://mblogthumb-phinf.pstatic.net/MjAyMzEwMjFfNTMg/MDAxNjk3ODQ1NDI4NDk5.lz4l9LSm-znlfXxHK4ekPLjWmkAEfp-xMFTEuwy_8iwg.OHD1H2cwkXQ7nZtETfjgdELxhKWxElNXjlCyVpwOWLsg.PNG.ghkdwjdtka/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2023-10-20_224405.png?type=w800
        st.markdown(user_input)

          # Add user message to chat history
        

    with st.chat_message("assistant", avatar="🐌"):
        # response = st.write_stream(response_generator())
        prompt = create_prompt(user_input)
        response = st.write_stream(generate_response(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("""
    <style>
    /*
   .stApp {
        background-color: lightblue;
    }   
    */    
    

    .stButton > button {
            color: white;
            background-color: black;
    }
                    
    
    h1 {
	    color: #FF3F00;
    }
            
    # .stTextInput {
    #         position:fixed;
    #         bottom: 3rem;

    # }
            
    .stChatMessage {
        justify-content: flex-start !important;
    }
            
    </style>
    """, unsafe_allow_html=True
)