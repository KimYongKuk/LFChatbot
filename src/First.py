import streamlit as st
from streamlit_extras.switch_page_button import switch_page 

st.set_page_config(page_title='L&F Chat ', page_icon=':🐳:', layout="wide")

def redirect_toMain():
     switch_page('Main')

print(st.query_params["user_name"])

redirect_toMain()

