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

load_dotenv()

# Set OpenAI API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get('OPENAI_API_KEY'),
)


def get_embedding(text, engine): # 입력받은 소스 텍스트와, 임베딩 모델을 통해 임베딩을 진행하는 함수.
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=engine).data[0].embedding




folder_path = './data'
# 폴더 및 모든 파일 중 txt파일만 리스트로 가져와서 for 문 시작
txt_file_name = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

data = []

for file in txt_file_name:
    txt_file_path = os.path.join(folder_path, file)
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        file_name = file + '.csv'
        file_path = os.path.join(folder_path, file_name)
        text = f.read()
        data.append(text)
        df = pd.DataFrame(data, columns=['text'])
        df['embedding'] = df.apply(lambda row: get_embedding(row.text, engine="text-embedding-ada-002"), axis=1)
        df.to_csv(file_path, index=False, encoding='utf-8')
    

# - embedding.csv파일을 만든다 zz
# - 첫번째 열은 'text'로 정의하고, 해당 열에는 각 파일의 정책 텍스트 내용이 들어간다
# - 두번째 열은 'embedding'으로 정의하고, 해당 열에는 위에서 선언한 get_embedding함수를 통해, 왼쪽에 있는 text내용을 임베딩한 값을 넣어준다. 각 text의 N차원 공간의 좌표가 기록된다.
# L&F는 ~~ 기여합니다 "," [-0.0199912931923....]

# 두 임베딩간 유사도 계산
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

# 질문을 임베딩하고, 유사도 높은 탑3 자료
def return_answer_candidate(df, query):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    # 입력된 질문과 각 문서의 유사도
    df['similarity'] = df['embedding'].apply(lambda x: cos_sim(np.array(query_embedding), np.array(x)))
    # 유사도 높은 순으로 정렬
    top3 = df.sort_values("similarity", ascending=False).head(3)
    return top3

# - 사용자의 입력질문과 57개 자료의 유사도를 측정해서, 가장 유사한 3개의 자료를 가져오는 함수
# - 사용자의 질문을 임베딩하여 1,536차원의 좌표를 얻는다
# - 57개 자료와 사용자 질문간 유사도를 측정한다
# df['similarity'] = df['embedding'].apply(lambda x: cos_sim(np.array(query_embedding), np.array(x)))
# - 가장 유사도가 높은 자료 3개를 가져온다
#  top3 = df.sort_values("similarity", ascending=False).head(3)



# 질문에 대한 가장 유사한 문서3개 가져와서, messages셋 만들어서 리턴
def create_prompt(df, query):
    
    # 질문과 가장 유사한 문서 3개 가져오기
    result = return_answer_candidate(df, query)
    print(result) # 결과 DataFrame 출력


    # 문서 내용을 동적으로 포맷팅하기 위한 코드
    system_message_parts = [
        f"문서{i + 1}: {text}"
        for i, text in enumerate(result['text'])  # result DataFrame에서 'text' 열의 모든 항목을 순회
    ]

    # 모든 문서 부분을 개행 문자로 구분하여 하나의 문자열로 합침
    documents_formatted = "\n".join(system_message_parts)

    # 최종 system_message에 문서 내용 포함
    system_message = f"""
    그리고 너의 이름은 L&F봇이야. 너는 L&F의 정책을 알려주기 위해 만들어진 챗봇이야.
    너는 주어진 문서를 참고해서, 자세하게 대답해줘.
    문서내용:
    {documents_formatted}
    """

    # system_message = f"""
    # 그리고 너의 이름은 L&F봇이야. 너는 L&F의 정책을 알려주기 위해 만들어진 챗봇이야.
    # 너는 주어진 문서를 참고해서, 자세하게 대답해줘.
    # 문서내용:
    # 문서1: """ + str(result.iloc[0]['text']) + """
    # """

    

    user_message = f"""User question: "{str(query)}". """

    messages =[
        {"role": "user", "content": user_message},
        {"role": "system", "content": system_message}
    ]
    print(result)
    return messages

# - 우리에게 익숙한 gpt에게 던질 messages작성하는 과정
# - 가장 유사한 자료 3개를 가져온다
# result = return_answer_candidate(df, query)
# - 가져온 3개의 자료의 내용을 system메세지에 적절하게 작성해준다
# - 유저가 입력한 질문을 더해준다
# - gpt/clova에 던질 최종 messages를 완성한다.


# 완성된 질문에 대한 답변 생성
def generate_response(messages):
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = messages,
        temperature = 1,
        max_tokens = 500)
    print(result.choices[0].message.content)
    return result.choices[0].message.content


st.title("WELCOME TO L&F WORLD") 
if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []



with st.form('form', clear_on_submit=True):
    user_input = st.text_input('안녕하세요 L&F 봇입니다. 회사 정책에 대해 궁금한 점이 있으시면 물어보세요!', '', key='input')
    submitted = st.form_submit_button('Send')
    

if submitted and user_input:
    # 프롬프트 생성 후 프롬프트를 기반으로 챗봇의 답변을 반환
    prompt = create_prompt(df, user_input)
    chatbot_response = generate_response(prompt)
    st.session_state["generated"].append(chatbot_response)
    st.session_state['past'].append(user_input)
    

if st.session_state['generated']:
    # 화면을 두 부분으로 나눔
   

    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="croodles")
        message(st.session_state["generated"][i], key=str(i), avatar_style="adventurer")
        
        
st.markdown("""
    <style>
    /*
   .stApp {
        background-color: lightblue;
    }   
    */    
    .
            
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