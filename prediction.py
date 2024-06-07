# prediction.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def app():
    st.title("Prediction Dashboard")
    
    # Add custom CSS
    st.markdown(
        """
        <style>
        .main {
            background-color: #000000;
            color: #ffffff;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .stDataFrame {
            border: 2px solid #ffffff;
            border-radius: 10px;
            overflow: hidden;
        }
        .stMarkdown {
            background-color: #333333;
            padding: 10px;
            border-radius: 10px;
            color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6, .stText {
            color: #ffffff;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.title("📊 Prediction Dashboard with Machine Learning")

    if 'dataframe' not in st.session_state:
        st.warning("Please load the data through the 'Google Sheet Link' menu first.")
        return
    
    df = st.session_state['dataframe']
    
    # Data Cleaning
    df['Sales'] = df['Sales'].astype(str).str.replace(',', '').astype(float)
    df['Profit'] = df['Profit'].astype(str).str.replace(',', '').astype(float)
    df['Discount'] = df['Discount'].astype(str).str.replace(',', '').astype(float)
    
    # Display the data
    st.subheader("Loaded Data")
    st.dataframe(df.head())

    # Data Preparation for Machine Learning
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('Order Month')['Sales'].sum().reset_index()

    # Splitting the data into training and test sets
    X = monthly_sales.index.values.reshape(-1, 1)
    y = monthly_sales['Sales'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Displaying the results
    st.subheader("📈 Linear Regression Model Results")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_train, y_train, color='blue', label='Training Data')
    ax.scatter(X_test, y_test, color='green', label='Test Data')
    ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
    ax.set_xlabel('Months')
    ax.set_ylabel('Sales')
    ax.set_title('Monthly Sales Prediction')
    ax.legend()
    st.pyplot(fig)

    # Model Evaluation
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"**Mean Squared Error:** {mse}")

    # Future Predictions
    st.subheader("📅 Future Sales Prediction")
    future_months = pd.date_range(start=df['Order Date'].max(), periods=12, freq='M').to_period('M').astype(str)
    future_indices = range(len(monthly_sales), len(monthly_sales) + len(future_months))
    future_sales = model.predict([[i] for i in future_indices])
    future_predictions = pd.DataFrame({'Order Month': future_months, 'Predicted Sales': future_sales})
    st.write(future_predictions)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(future_months, future_sales, color='purple', linewidth=2, label='Future Predictions')
    ax.set_xlabel('Months')
    ax.set_ylabel('Predicted Sales')
    ax.set_title('Future Monthly Sales Prediction')
    ax.legend()
    st.pyplot(fig)

    # High Sales Products
    st.header("🏆 High Sales Products")
    high_sales_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(high_sales_products)
    
    # Profit Margins
    st.header("💹 Profit Margins")
    profit_margins = df.groupby('Product Name')['Profit'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(profit_margins)
    
    # High-Value Customers
    st.header("💎 High-Value Customers")
    high_value_customers = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(high_value_customers)
    
    # Discount Impact on Profit
    st.header("🔍 Discount Impact on Profit")
    discount_profit = df.groupby('Discount')['Profit'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(discount_profit['Discount'], discount_profit['Profit'], marker='o')
    ax.set_title('Impact of Discount on Profit')
    ax.set_xlabel('Discount')
    ax.set_ylabel('Profit')
    st.pyplot(fig)
    
    # Shipping Efficiency
    st.header("🚚 Shipping Efficiency")
    shipping_efficiency = df.groupby('Ship Mode')['Profit'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(shipping_efficiency['Ship Mode'], shipping_efficiency['Profit'], color='skyblue')
    ax.set_title('Shipping Mode Efficiency')
    ax.set_xlabel('Ship Mode')
    ax.set_ylabel('Profit')
    st.pyplot(fig)
    
    # Credit Risk Analysis
    st.header("⚠️ Credit Risk Analysis")
    credit_risk_df = df.groupby('Customer Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Discount': 'mean'
    }).reset_index()
    
    # Identify high-risk customers
    high_risk_customers = credit_risk_df[(credit_risk_df['Profit'] < 0) & (credit_risk_df['Discount'] > 0.2)]
    st.subheader("High-Risk Customers")
    st.write(high_risk_customers)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(high_risk_customers['Sales'], high_risk_customers['Profit'], color='red')
    ax.set_title('High-Risk Customers: Sales vs. Profit')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Profit')
    st.pyplot(fig)

if __name__ == '__main__':
    app()
