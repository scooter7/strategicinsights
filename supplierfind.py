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

def query_csv_with_gpt(prompt, df_chunk):
    context = df_chunk.to_csv(index=False)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Using the following CSV data chunk:\n\n{context}\n\nAnswer the following question: {prompt}"}
        ],
        max_tokens=150,
        temperature=0.5,
    )
    return response.choices[0].message['content'].strip()

def clean_response(response):
    lines = response.split("\n")
    clean_lines = [line.strip() for line in lines if "Company" in line and "annual sales" in line]
    return "\n".join(clean_lines)

def aggregate_responses(responses):
    companies = set()
    for response in responses:
        cleaned_response = clean_response(response)
        for line in cleaned_response.split("\n"):
            companies.add(line)
    return "\n".join(sorted(companies))

# Streamlit app UI
st.title("Conversational CSV Query App")
st.write("This app allows you to query a CSV file hosted on GitHub conversationally using OpenAI's GPT-3.5-turbo.")

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
                    response = query_csv_with_gpt(user_query, chunk)
                    if response:  # Check if the response is not empty
                        responses.append(response)
                
                aggregated_response = aggregate_responses(responses)
                if aggregated_response:
                    st.write("Response:")
                    st.write(aggregated_response)
                else:
                    st.write("No relevant data found in the provided CSV file.")
        else:
            st.error("Please enter a question.")
