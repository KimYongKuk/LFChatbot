from Constants import *
from VectorDB import VectorDB
# from Login import *
import streamlit_authenticator as stauth
import os
import pandas as pd
import numpy as np
import time

from numpy import dot
from numpy.linalg import norm
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from ollama import Client
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
from openai import OpenAI

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws.llms.bedrock import BedrockLLM
# from langchain_community.llms.bedrock import BedrockLLM
from langchain_aws import ChatBedrock
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# region 함수정의

# def init_state(param):
#     if param == 'Login':
#         if 'authentication_status' not in st.session_state:
#             st.session_state['authentication_status'] = None

#         if not st.session_state.get('logout'):
#             st.session_state['logout'] = None

#         if not st.session_state.get('name'):
#             st.session_state['name'] = None

#         if not st.session_state.get('name'):
#             st.session_state['name'] = None

#         if not st.session_state.get('username'):
#             st.session_state['username'] = None


# 프롬프트 생성
def create_prompt(query):
    
    # 질문과 가장 유사한 문서 3개 가져오기
    result = vector_db.answer_vectorDB(query)
    # print('###데이터프레임###')
    # print(result) # 결과 DataFrame 출력

    # 문서 내용을 동적으로 포맷팅하기 위한 코드
    system_message_parts = []
    system_message_parts.extend(doc.page_content for doc in result)

    # 모든 문서 부분을 개행 문자로 구분하여 하나의 문자열로 합침
    documents_formatted = "\n\n\n".join(system_message_parts) + "\n\n\n"

   

    # 최종 system_message에 문서 내용 포함
    system_message = f"""
    나는 L&F봇입니다. 나는 L&F의 사내정책, 문화, 조직에 대해 알려주기 위해 만들어진 챗봇입니다.
    아래 세 개의 역따옴표로 구분된 텍스트는 사내 문서입니다.
    ```{documents_formatted}```
    1. user question 내용이 사내 문서에 없다면, '죄송합니다. 그 부분은 모르겠습니다.' 라고만 답해야 합니다.
    2. 사내 문서에 근거하여 간결하게 답하세요.
    3. 링크가 있다면 링크를 제공하세요.
    4. 무조건 한글로 대답하세요.
    """

    print('###문서 포맷###')
    print(system_message)

    user_message = f"""User question: "{str(query)}". """

    messages =[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # messages =[
    #     SystemMessage(
    #         content=system_message
    #     ),
    #     HumanMessage(
    #         content=user_message
    #     )
    # ]
    return messages

# 프롬프트 생성 (bedrock normal)
def create_prompt_bedrock(query):

    # 최종 system_message에 문서 내용 포함
    # system_message = f"""
    # 1. 무조건 한글로 대답하세요.
    # 2. 친절하고, 간결하게 대답하세요.
    # 3. 모르는 건 모른다고 명확하게 대답하세요.
    # """

    return str(query)

# 프롬프트 생성 (일반모드)
def create_prompt_normal(query):

    # 최종 system_message에 문서 내용 포함
    # system_message = f"""
    # 1. 무조건 한글로 대답하세요.
    # 2. 친절하고, 간결하게 대답하세요.
    # 3. 모르는 건 모른다고 명확하게 대답하세요.
    # """

    user_message = f"""User question: "{str(query)}". """

    messages =[
        {"role": "system", "content": "당신은 한글챗봇입니다."},
        {"role": "user", "content": user_message}
    ]

    return messages

# 완성된 질문에 대한 답변 생성
def generate_response(messageArr):
    print('하히하호1')
    print(messageArr)
    print('하히하호2')
    # #lm studio
    # client = OpenAI(base_url="http://192.168.120.253:1234/v1", api_key="lm-studio")
    
    # print(messageArr)
    
    # response = client.chat.completions.create(
    #     model="heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF",
    #     messages=messageArr,
    #     temperature=0.7,
    #     stream=True
    # )

    bedrock_llm = ChatBedrock(
        credentials_profile_name='default',
        # model_id='anthropic.claude-3-haiku-20240307-v1:0',
        model_id='anthropic.claude-3-haiku-20240307-v1:0',
        # model_id='anthropic.claude-3-sonnet-20240229-v1:0',
        model_kwargs= {
            # "prompt": "\n\nHuman:<prompt>\n\nAssistant:",
            "temperature":0.3
        },
        streaming=True,
        # callbacks=[StreamingStdOutCallbackHandler()],
        region_name='us-east-1'
        # region_name='ap-southeast-2'
    )


    # sAnswer = bedrock_llm.stream(messageArr)

    for chunk in bedrock_llm.stream(messageArr):
        yield chunk.content
        # time.sleep(0.01)

    # for chunk in sAnswer:
    #     # print(chunk['message']['content'], end='', flush=True)
    #     token = chunk['message']['content']
    #     res_message += token
    #     yield chunk['message']['content']
    # sAnswer = bedrock_llm.predict(messageArr)


    # print(sAnswer)
    # return sAnswer
    
# endregion




# region Top,Sidebar Layout
## Top   


# init_state('Login')

# if st.session_state["authentication_status"] == None:
    # authenticator.login()

# if st.session_state["authentication_status"]:
st.set_page_config(page_title='L&F Chat ', page_icon=':🐳:', layout="wide")
load_dotenv()

# VectorDB 인스턴스 생성
vector_db = VectorDB()
# vector_db.perClient.reset()

# print(st.query_params["user_name"])

# st.experimental_set_query_params()
st.error('🚨 검색어에 주의하세요, 불필요한 검색은 삼가해주세요!')
chatMode = st.radio(key='chatMode', label = 'chat mode', label_visibility="hidden", options = ['Normal', 'Confidential']) 
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
st.session_state.checkChatMode = chatMode
st.session_state.checkbtn = 'uncheck'

colored_title = """
    <h1 style="margin-top:20px;">
    <span style="color: white;">WELCOME TO</span><span style="color:#FF3F00;">&nbsp; L&F BOT🔥</span>
</h1>
"""
if "chatMode" not in st.session_state:
    st.session_state["chatMode"] = "Normal"

if "colored_title" not in st.session_state:
    st.session_state["colored_title"] = colored_title

# else:
#     st.snow()

st.markdown(st.session_state["colored_title"], unsafe_allow_html=True)
st.logo("https://www.landf.co.kr/img/mlogo.png", icon_image="https://www.landf.co.kr/img/mlogo.png")

## Side Bar
with st.sidebar:
    # authenticator.logout()

    # st.success(st.query_params["user_name"] + '님 안녕하세요! 🥰')
    st.warning('🔥 주의사항에 대한 안내를 확인하세요. http://lfon.landf.co.kr')
    st.markdown('---')
    # st.success('🔥 아래는 검색어 예시입니다.')
    st.write('<p style="font-size: 17px;"><span class="bold">🤗아래는 검색어 예시입니다🤗</span></p> ', unsafe_allow_html=True)

    # 프롬프트 예시
    st.markdown('<p style="font-size: 15px;"> # MES 담당자가 누구야? </p>\
                <p style="font-size: 15px;"> # LFON 계정을 생성하고 싶어. </p>\
                <p style="font-size: 15px;"> # 포맷 하고싶은데 어떻게해야해? </p>\
                <p style="font-size: 15px;"> # L&F의 조직문화에 대해 알려줘. </p>\
                <p style="font-size: 15px;"> # 전표 프로세스가 궁금한데 알려줄래? </p>', unsafe_allow_html=True)
    
    # DB UPDATE
    st.markdown('---')
    # st.write('<p style="font-size: 17px;"><span class="bold">DB UPDATE </span></p> ', unsafe_allow_html=True)
    emBtnResult = st.button("DB UPDATE", type='primary')
    if emBtnResult:
        with st.spinner('업데이트 중입니다. 잠시만 기다려주세요.'):
            chkVal = vector_db.create_embedding()
            st.session_state.checkbtn = 'check'
        
        if chkVal == 's':
            st.success('업데이트 완료!')

        else:
            st.error('오류가 발생했습니다. 오류내용 : ' + chkVal, icon="🚨")

if st.session_state.checkbtn != 'check':
    if st.session_state.checkChatMode == "Normal":
        st.toast('🎉일반채팅 ON🎉')

    else:
        st.toast('🔥보안채팅 ON🔥')




# elif st.session_state["authentication_status"] is False:
#     st.error('UserName/Password is incorrect.')

# elif st.session_state["authentication_status"] is None:
#     st.warning('Please enter your username and password')


# endregion


    
# region chat logic

# Initialize chat history
# if st.session_state["authentication_status"]:
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "https://media.bunjang.co.kr/product/254293734_1_1708650713_w360.jpg" if message["role"] == "user" else "🐌"

    with st.chat_message(message["role"], avatar = avatar):
        st.markdown(message["content"])

if user_input := st.chat_input("'LFON 계정생성을 하고싶어', '경영정보팀에 대해 알려줘'"):

    with st.chat_message("user", avatar="https://media.bunjang.co.kr/product/254293734_1_1708650713_w360.jpg"):
        st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        

    with st.chat_message("assistant", avatar="🐌"):

        if st.session_state.checkChatMode == "Normal":
            prompt = create_prompt_bedrock(user_input)
        else: 
            prompt = create_prompt(user_input)

        response = st.write_stream(generate_response(prompt))
        # response = st.write(generate_response(prompt))
        # print('###사용자 질의###')    
        # print(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})



# endregion

