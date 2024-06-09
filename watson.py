import streamlit as st
import numpy as np
import pandas as pd
from millify import millify
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
import altair as alt

def app():
  # Konfigurasi IBM Watson API
        API_KEY = 'O02u_evhjylXQYdVJf5GuySdjk5UJq5ZtLU11rN6xz1p'
        WML_AUTH_ENDPOINT = 'https://iam.cloud.ibm.com/identity/token'
        WML_ENDPOINT = 'https://jp-tok.ml.cloud.ibm.com/ml/v4/deployments/capstone/predictions?version=2021-05-01'
        
        # Fungsi untuk mendapatkan akses token
        def get_access_token(api_key):
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            data = f'apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey'
            response = requests.post(WML_AUTH_ENDPOINT, headers=headers, data=data)
            if response.status_code == 200:
                return response.json()['access_token']
            else:
                st.error(f"Failed to get access token. Status code: {response.status_code}, Response: {response.text}")
                return None
        
        # Fungsi untuk memanggil API IBM Watson
        def call_watson_ml(payload, access_token):
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }
            response = requests.post(WML_ENDPOINT, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to get prediction. Status code: {response.status_code}, Response: {response.text}")
                return None
        # Initialize session state to store data
        if 'data' not in st.session_state:
            st.session_state.data = []
        
        # Display the data
        st.subheader("Loaded Data")
        if st.session_state.data:
            df = pd.DataFrame(st.session_state.data, columns=["Day", "Month", "Year", "Revenue", "Cost", "Profit", "Profit %"])
            st.dataframe(df)
        else:
            st.write("No data available.")    

if __name__ == '__main__':
    app()
