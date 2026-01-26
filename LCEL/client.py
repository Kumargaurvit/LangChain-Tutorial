import streamlit as st
import requests

def get_groq_response(input_text: str, language: str):
    json_body = {
        "input" : {
            "language" : language,
            "text" : input_text
        },
        "config" : {},
        "kwargs" : {}
    }

    response = requests.post("http://127.0.0.1:8000/chain/invoke",json=json_body)

    try:
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        return e

st.title("LLM App using LangChain")
input_text = st.text_input("Ask what's on your mind")
language = st.sidebar.text_input("Enter the language you want the answer in (English is default)")

if input_text:
    if language:
        answer = get_groq_response(input_text=input_text, language=language)["output"]
        st.write(answer)
    else:
        answer = get_groq_response(input_text=input_text, language="English")["output"]
        st.write(answer)
else:
    st.info("Please enter text to generate answer!")