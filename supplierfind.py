import streamlit as st
import pandas as pd
import openai
import requests
import io

# Set your OpenAI API key using Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

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

def summarize_csv(df):
    summary = ""
    for col in df.columns:
        summary += f"Column '{col}' has {df[col].nunique()} unique values, some examples: {df[col].dropna().unique()[:3].tolist()}\n"
    return summary

def query_csv_with_gpt(prompt, df_summary):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Using the following CSV data summary:\n\n{df_summary}\n\nAnswer the following question: {prompt}"}
        ],
        max_tokens=150,
        temperature=0.5,
    )
    return response.choices[0].message['content'].strip()

# Streamlit app UI
st.title("Conversational CSV Query App")
st.write("This app allows you to query a CSV file hosted on GitHub conversationally using OpenAI's GPT-4.")

github_url = "https://github.com/scooter7/strategicinsights/blob/main/docs/csv_data.csv"
st.write(f"Fetching data from: {github_url}")

# Load data
df = load_data_from_github(github_url)

if df is not None:
    st.write("CSV Data Preview:")
    st.dataframe(df.head())

    # Generate summary
    df_summary = summarize_csv(df)
    
    # User query input
    user_query = st.text_input("Enter your question about the CSV data:")

    if st.button("Submit"):
        if user_query:
            with st.spinner("Generating response..."):
                answer = query_csv_with_gpt(user_query, df_summary)
                st.write("Response:")
                st.write(answer)
        else:
            st.error("Please enter a question.")
