import os
import streamlit as st
import pandas as pd
import requests
import io
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=google_api_key)

@st.cache_data
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.content.decode('utf-8', errors='ignore')
        try:
            df = pd.read_csv(io.StringIO(data), on_bad_lines='skip')
            return df
        except pd.errors.ParserError as e:
            st.error(f"Parser error: {e}")
            return None
    else:
        st.error("Error loading the CSV file from GitHub")
        return None

def clean_dataframe(df):
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip().replace({'\n': ' ', '\r': ' '}, regex=True)
    return df

def chunk_df(df, chunk_size=100):
    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def query_csv_with_google(prompt, df_chunk):
    context = df_chunk.to_csv(index=False)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Using the following CSV data chunk:\n\n{context}\n\nAnswer the following question: {prompt}"}
    ]
    response = genai.generate(messages)
    return response['choices'][0]['message']['content'].strip()

def aggregate_responses(responses, prompt):
    combined_response = ""
    relevant_info = set()
    for response in responses:
        for line in response.split("\n"):
            if line.strip():
                relevant_info.add(line.strip())
    if relevant_info:
        combined_response = "\n".join(sorted(relevant_info))
    else:
        combined_response = "No relevant data found in the provided CSV file."
    return combined_response

# Streamlit app UI
st.title("Conversational CSV Query App")
st.write("This app allows you to query a CSV file hosted on GitHub conversationally using Google Generative AI.")

github_url = "https://raw.githubusercontent.com/scooter7/strategicinsights/main/docs/csv_data.csv"
st.write(f"Fetching data from: {github_url}")

# Load data
df = load_data_from_github(github_url)

if df is not None:
    df = clean_dataframe(df)
    st.write("CSV Data Preview:")
    st.dataframe(df.head())

    # User query input
    user_query = st.text_input("Enter your question about the CSV data:")

    if st.button("Submit"):
        if user_query:
            with st.spinner("Processing data..."):
                df_chunks = chunk_df(df)
                responses = []
                
                for chunk in df_chunks:
                    response = query_csv_with_google(user_query, chunk)
                    if response:  # Check if the response is not empty
                        responses.append(response)
                
                aggregated_response = aggregate_responses(responses, user_query)
                st.write("Response:")
                st.write(aggregated_response)
        else:
            st.error("Please enter a question.")
