import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
# import ast
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
# import time
import streamlit.components.v1 as components
# import random
# import asyncio
# import logging
# import requests 
# import json
# import ollama
from ollama import Client
# import asyncio
# from ollama import AsyncClient
# import requests
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title='L&F Chat ', page_icon=':🐳:', layout="wide")

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

with st.sidebar:
    # with st.echo():
    #     st.write("사이드바")

    # with st.spinner("Loading..."):
    #     time.sleep(5)
    st.success('주의사항에 대한 안내를 확인하세요. http://lfon.landf.co.kr')
    st.success('🔥 아래는 검색어 예시입니다.')
    st.write('<br>[1] mes 담당자가 누구야?<br><br>[2] hr 담당자가 누구야? <br><br> [3] LFON 계정을 생성하고 싶어.', unsafe_allow_html=True)


NAME = '김용국(wg0403)' #CONST
res_message = 'test'

# Set OpenAI API key
clientEmbedding = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
)

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
    # 입력된 질문과 각 문서의 유사도
    df['similarity'] = df['embedding'].apply(lambda x: cos_sim(np.array(query_embedding), np.array(x)))
    # 유사도 높은 순으로 정렬
    top2 = df.sort_values("similarity", ascending=False).head(3)
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
    print('###데이터프레임###')
    print(result) # 결과 DataFrame 출력

    
    # 문서 내용을 동적으로 포맷팅하기 위한 코드
    system_message_parts = [
        f"문서{i + 1}: {text}"
        for i, text in enumerate(result['text'])  # result DataFrame에서 'text' 열의 모든 항목을 순회
    ]

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
    4. 무조건 존댓말로 답하세요.
    """

    # system_message = f"""
    # 너의 이름은 L&F봇이야. 너는 L&F의 사내정책, 문화, 조직에 대해 알려주기 위해 만들어진 챗봇이야.
    # 너는 주어진 문서만을 참고해서 대답해줘. 질문과 문서와 관련없는 내용은 절대 답변하지마.
    # """

    # system_message =f"""
    # 너의 이름은 L&F봇이야. 너는 L&F의 사내정책, 문화, 조직에 대해 알려주기 위해 만들어진 챗봇이야.
    # 한글로만 대답해줘.
    # """
    
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
    response = client.chat(model='EEVE-10.8B', messages=messageArr,
        stream = True,
        options = {
            "temperature": 0,
            "top_k":10,
            "top_p":0.5,
            "num_ctx":500,
            "mirostat_tau":1.0,
            "mirostat":1
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

#### main logic


# React to user input
if user_input := st.chat_input("'LFON 계정생성을 하고싶어', '경영정보팀에 대해 알려줘'"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="https://media.bunjang.co.kr/product/254293734_1_1708650713_w360.jpg"):
        st.markdown(user_input)
        
    with st.chat_message("assistant", avatar="🐌"):
        prompt = create_prompt(user_input)
        response = st.write_stream(generate_response(prompt))
        print('###사용자 질의###')
        print(user_input)

    st.session_state.messages.append({"role": "assistant", "content": message})


