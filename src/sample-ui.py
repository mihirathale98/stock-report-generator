import streamlit as st
import requests
import asyncio
import aiohttp


# from openai_api import OpenAI

def mock_main(query):
    return ["This is a mock answer", "Mock transcipts :red[blah blah blah]"]


st.set_page_config(page_title="Demo", page_icon="📖", layout="wide")

st.title("Analysis of earnings call transcripts")

user_input = st.text_area('Enter your question here...')
company_name = st.text_input('Enter your company name here...')
quarter = st.selectbox(
    'Select the quarter',
    ('Q1', 'Q2', 'Q3', 'Q4'))

year = st.selectbox(
    'Select the year',
    ('2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'))

generate = st.button("Generate")

if generate:
    with st.spinner("This might take a while..."):
        outputs = (mock_main(user_input))

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**:Actual Answer**")
                st.write(outputs[0])
            with col2:
                st.write(f"**:Readings from transcipts**")
                st.markdown(outputs[1])
