from Auth import *
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

hashedPasswords = Hasher(['abc']).generate()

with open('config.yaml','r', encoding="utf-8") as file:
    data = yaml.safe_load(file)

if 'password' in data:
     data['password'] = Hasher(data['password']).generate()

with open('config.yaml','w', encoding="utf-8") as file:
    yaml.dump(data, file, default_flow_style=False)

with open('config.yaml', encoding="utf-8") as file:
    config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )