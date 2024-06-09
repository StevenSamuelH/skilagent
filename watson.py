import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests
import json
import plotly.graph_objects as go

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

def app():
    # Initialize session state to store data
    if 'data' not in st.session_state:
        st.session_state.data = []

  
    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data, columns=["Day", "Month", "Year", "Revenue", "Cost", "Profit", "Profit %"])
        st.dataframe(df)
    

    # Prediksi kategori profit dengan model dari IBM Watson
    st.header("🔮 Prediction of Profit Category UMKMAI with IBM Watson ML")

    # Convert float inputs to double precision
    # choose day
    day_options = list(range(1, 32))  
    day = st.selectbox("Select the day:", day_options)

    # choose month 
    month_options = list(range(1, 13)) 
    month = st.selectbox("Select the month:", month_options)
    # choose year
    year_options = [2012, 2014, 2016, 2018, 2022, 2023]
    year = st.selectbox("Select the year:", year_options)
    revenue = np.float64(st.number_input("Enter the revenue:"))
    cost = np.float64(st.number_input("Enter the cost:"))
    profit = np.float64(st.number_input("Enter the profit:"))
    profit_percentage = np.float64(st.number_input("Enter the profit percentage:"))

    if st.button("Predict Profit Category with IBM Watson ML"):
        access_token = get_access_token(API_KEY)
        if access_token:
            payload = {
                "input_data": [
                    {
                        "fields": ["Month", "Day", "Year", "Revenue", "Cost", "Profit", "Profit %"],
                        "values": [[month, day, year, revenue, cost, profit, profit_percentage]]
                    }
                ]
            }
            st.write(f"Payload: {payload}")  # Debug statement

            result = call_watson_ml(payload, access_token)
            if result:
                st.write(f"Result: {result}")  # Debug statement
                try:
                    prediction = result['predictions'][0]['values'][0][0]  # Extracting the predicted category directly
                    categories = ["very bad", "bad", "fair", "good", "excellent"]

                    # Define colors for each category
                    colors = ["darkred", "red", "yellow", "lightgreen", "green"]

                    # Create a bar chart
                    fig = go.Figure()

                    # Create bars for each category
                    for i, category in enumerate(categories):
                        if category == prediction:
                            bar_length = 100  # Set the bar length to 100 for predicted category
                        else:
                            bar_length = 0
                        fig.add_trace(go.Bar(x=[bar_length], y=[category], text=[f"{bar_length:.2f}%"], textposition='auto',
                                            marker_color=colors[i], orientation='h'))

                    fig.update_layout(title='Predicted Profit Category', xaxis_title='Probability', yaxis=dict(showticklabels=True))
                    st.plotly_chart(fig)

                    # Display the predicted category
                    st.write(f"Predicted Profit Category: {prediction.capitalize()}")

                    # Kotak solusi
                    st.subheader("Solusi Keuangan")
                    if prediction == "very bad":
                        st.write("Situasi keuangan Anda terlihat kritis. Pertimbangkan untuk berkonsultasi dengan ahli keuangan untuk membuat rencana mengelola utang dan pengeluaran.")
                        st.write("Beberapa langkah yang dapat Anda ambil:")
                        st.write("- Evaluasi Utang Anda dan pertimbangkan untuk konsolidasinya jika memungkinkan.")
                        st.write("- Fokuskan pengeluaran pada kebutuhan mendesak dan hindari pengeluaran yang tidak perlu.")
                        st.write("- Cari bantuan dari ahli keuangan untuk mendapatkan saran yang sesuai dengan situasi Anda.")
                    elif prediction == "bad":
                        st.write("Situasi keuangan Anda tidak optimal. Pertimbangkan untuk membuat anggaran dan mengurangi pengeluaran yang tidak perlu untuk meningkatkan kesehatan keuangan Anda.")
                        st.write("Langkah-langkah yang dapat Anda ambil:")
                        st.write("- Evaluasi kembali anggaran bulanan Anda dan identifikasi area penghematan.")
                        st.write("- Fokuskan sumber daya pada kebutuhan mendesak dan hindari pengeluaran yang tidak perlu.")
                        st.write("- Cari peluang untuk meningkatkan pendapatan Anda.")
                    elif prediction == "fair":
                        st.write("Situasi keuangan Anda rata-rata. Pertimbangkan untuk menabung lebih banyak dan berinvestasi dengan bijak untuk mengamankan masa depan keuangan Anda.")
                        st.write("Beberapa langkah yang bisa Anda ambil:")
                        st.write("- Tetapkan tujuan tabungan jangka pendek dan jangka panjang, dan sisihkan sebagian dari pendapatan Anda untuk mencapainya.")
                        st.write("- Pelajari berbagai pilihan investasi yang tersedia dan pilihlah yang sesuai dengan profil risiko dan tujuan keuangan Anda.")
                        st.write("- Tinjau dan perbarui rencana keuangan jangka panjang Anda secara berkala.")
                    elif prediction == "good":
                        st.write("Situasi keuangan Anda baik. Lanjutkan menabung dan berinvestasi dengan bijak untuk mempertahankan dan meningkatkan kesehatan keuangan Anda.")
                        st.write("Langkah-langkah yang dapat Anda ambil:")
                        st.write("- Pertimbangkan untuk mendiversifikasi portofolio investasi Anda untuk mengurangi risiko.")
                        st.write("- Tinjau kembali rencana keuangan jangka panjang Anda dan pertimbangkan untuk meningkatkan investasi sesuai dengan tujuan Anda.")
                    elif prediction == "excellent":
                        st.write("Selamat! Situasi keuangan Anda sangat baik. Lanjutkan kinerja bagus Anda dan pertimbangkan untuk menjelajahi lebih banyak peluang investasi.")
                        st.write("Beberapa langkah yang bisa Anda ambil:")
                        st.write("- Telusuri berbagai pilihan investasi yang tersedia untuk memperluas portofolio Anda.")
                        st.write("- Jika belum melakukannya, pertimbangkan untuk merencanakan pensiun Anda dan alokasikan dana tambahan untuk tujuan ini.")
                    else:
                        st.write("Tidak ada solusi keuangan yang tersedia.")
                except KeyError as e:
                    st.error(f"Error extracting prediction: {e}")
                    st.error(f"Result: {result}")  # Debug statement

if __name__ == '__main__':
    app()
