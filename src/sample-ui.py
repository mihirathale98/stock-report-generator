import streamlit as st
import requests
from OpenAI import OpenAI
import nltk

"""
This is a sample UI for the Stock Report Generator app using Streamlit and FastAPI
"""

# Initialize the OpenAI API
openai_api = OpenAI(api_key="YOUR_KEY", model_name="gpt-3.5-turbo")

# API endpoints for retrieval and QA
retriever_url = "http://127.0.0.1:8002/retrieve"
url = 'http://127.0.0.1:8001/qa'

# Custom prompt
prepend_instruction = '''Instruction :Based on the query asked by the user please follow the following procedure to solve it:-
1. Identify what insight the user is looking for
2. Identify the relevant information from the given context
3. Generate a response such that it is based only on the given context
4. The response should help user to make informed decisions
5. Make use of numeric values and figures present in the context if they help in answering the question asked
6. Do not add any additional information'''

additional_instruction = '''Additional Instruction: You are an AI ChatBot which provides useful insights to user based on earnings call conversations.'''


def get_answer(url, query, context, max_new_tokens=512):
    '''
    Get response from custom LLM
    
    Args:
        url (str): API endpoint
        query (str): Query asked by the user
        context (str): Context from the transcript
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        response (str): Generated response
        
    '''
    model_input = '\n\n'.join([
        "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:",
        prepend_instruction, additional_instruction, "Context:", context, query, "### Response:"])
    response = requests.post(url, json={'query': model_input, 'max_new_tokens': max_new_tokens})
    return response.json()['answer'].split('<s>')[1].replace(model_input, '')


def get_openai_answer(query, context, max_new_tokens):
    '''
    Get response from OpenAI
    
    Args:
        query (str): Query asked by the user
        context (str): Context from the transcript
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        answer (str): Generated response
    '''
    model_input = '\n\n'.join([
        prepend_instruction, additional_instruction, context, query
    ])
    answer = openai_api.generate(model_input, max_new_tokens=max_new_tokens)
    return answer

# Streamlit UI code for the app with sidebar for customizing the parameters for the model and the API endpoints for retrieval and QA
st.set_page_config(page_title="Stock Report Generator", page_icon="📖", layout="wide")

st.title("Stock Report Generator")

user_input = st.text_area('Enter your question here...')
generate = st.button("Generate")

with st.sidebar:
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 512)
    use_context = st.checkbox("Use context with OpenAI")

if generate:
    with st.spinner("This might take a while..."):
        context = requests.post(retriever_url, json={'query': user_input}).json()['context']
        output_1 = get_answer(url, user_input, context, max_new_tokens)
        # output_1 = ""
        if use_context:
            output_2 = get_openai_answer(user_input, f'Context:\n{context}', max_new_tokens)
        else:
            output_2 = get_openai_answer(user_input, "", max_new_tokens)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**:green[Llama2-7b-chat]**")
                st.write(output_1.replace('\n', '\n\n'))
            with col2:
                st.write(f"**:red[GPT3.5-turbo]**")
                st.write(output_2.replace('\n', '\n\n'))

        with st.container():
            for cont in context.split('\n'):
                if (cont != ""):
                    with st.expander(nltk.sent_tokenize(cont)[0], expanded=False):
                        st.write(cont)

