import streamlit as st
import requests
import pandas as pd
from io import StringIO
import chardet

def app():
    st.title("Google Sheet Link Input")
    
    google_sheet_url = st.text_input("Enter Google Sheet link:")
    
    if google_sheet_url:
        try:
            csv_url = google_sheet_url.replace('edit?usp=sharing', 'gviz/tq?tqx=out:csv&sheet=Sheet1')
            response = requests.get(csv_url)
            
            # Check if the response content type is HTML (indicating an error page)
            if 'text/html' in response.headers.get('Content-Type', ''):
                st.error("The provided link is not a valid CSV link. Please ensure it follows the correct format.")
                return
            
            encoding = chardet.detect(response.content)['encoding']
            csv_data = response.content.decode(encoding)
            
            # Try to read the CSV data with different delimiters
            try:
                df = pd.read_csv(StringIO(csv_data))
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(StringIO(csv_data), delimiter=',')
                except pd.errors.ParserError:
                    st.error("Error parsing CSV data. Please check the CSV format.")
                    return
            
            # Display the first few rows and column names for debugging purposes
            st.write("First few rows of the loaded data:", df.head())
            st.write("Columns in the loaded data:", df.columns.tolist())
            # Convert columns to appropriate data types
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
            df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
            # df = df.dropna(subset=['Order Date', 'Ship Date'])
            df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
            df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
            # df['Order ID'] = df['Order ID'].astype(str)
            df['year'] = df['Order Date'].dt.year
            df['days to ship'] = abs(df['Ship Date'] - df['Order Date']).dt.days
            
            st.session_state['dataframe'] = df
            st.success("Data loaded successfully! Go to the Dashboard to see the visualizations.")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.error("Please ensure the Google Sheet link is correct and the data is formatted as a CSV.")
