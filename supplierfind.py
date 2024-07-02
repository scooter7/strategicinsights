import streamlit as st
import pandas as pd
import requests
import io
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your Google API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_data
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.content.decode('utf-8')
        try:
            df = pd.read_csv(io.StringIO(data), on_bad_lines='skip')
            return df
        except pd.errors.ParserError as e:
            st.error(f"Parser error: {e}")
            return None
    else:
        st.error("Error loading the CSV file from GitHub")
        return None

def chunk_df(df, chunk_size=100):
    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def query_csv_with_google(prompt, df_chunk):
    context = df_chunk.to_csv(index=False)
    messages = [
        {"content": "You are a helpful assistant."},
        {"content": f"Using the following CSV data chunk:\n\n{context}\n\nAnswer the following question: {prompt}"}
    ]
    response = genai.generate(messages=messages)
    return response.messages[0].content.strip()

def aggregate_responses(responses, prompt):
    unique_responses = set(responses)
    return f"Here is the consolidated answer for the prompt '{prompt}':\n" + "\n".join(unique_responses)

# Streamlit app UI
st.title("Conversational CSV Query App")
st.write("This app allows you to query a CSV file hosted on GitHub conversationally using Google Gemini.")

github_url = "https://raw.githubusercontent.com/scooter7/strategicinsights/main/docs/csv_data.csv"
st.write(f"Fetching data from: {github_url}")

# Load data
df = load_data_from_github(github_url)

if df is not None:
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
