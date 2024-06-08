import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
    df.drop_duplicates(inplace=True)
    df['Sales'] = df['Sales'].astype(str).str.replace(',', '').astype(float)
    df['Profit'] = df['Profit'].astype(str).str.replace(',', '').astype(float)
    df['Discount'] = df['Discount'].astype(str).str.replace(',', '').astype(float)
    
    df['Sales'].fillna(df['Sales'].median(), inplace=True)
    df['Profit'].fillna(df['Profit'].median(), inplace=True)
    df['Discount'].fillna(df['Discount'].median(), inplace=True)

    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Sales'] >= lower_bound) & (df['Sales'] <= upper_bound)]

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)
    df['Month'] = df['Order Date'].dt.month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Aggregate sales data by month
    monthly_sales = df.groupby('Order Month').agg({
        'Sales': 'sum',
        'Discount': 'mean',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Month_sin': 'first',
        'Month_cos': 'first'
    }).reset_index()

    monthly_sales['Lag_Sales'] = monthly_sales['Sales'].shift(1).fillna(0)
    X = monthly_sales[['Month_sin', 'Month_cos', 'Lagged_Sales', 'Discount', 'Profit', 'Quantity']].values
    y = monthly_sales['Sales'].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter tuning for Gradient Boosting Regressor
    param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10]
    }

    random_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_distributions=param_dist, n_iter=50, cv=5, scoring='r2', random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)

    st.subheader("📈 Gradient Boosting Model Results")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_train.index, y_train, color='blue', label='Training Data')
    ax.scatter(X_test.index, y_test, color='green', label='Test Data')
    ax.plot(X_test.index, y_pred, color='red', linewidth=2, label='Predicted Line')
    ax.set_xlabel('Months')
    ax.set_ylabel('Sales')
    ax.set_title('Monthly Sales Prediction')
    ax.legend()
    st.pyplot(fig)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"**Mean Squared Error:** {mse}")
    st.write(f"**R2 Score:** {r2}")
    st.write(f"**Mean Absolute Error:** {mae}")

    st.subheader("📅 Future Sales Prediction")
    future_months = pd.date_range(start=df['Order Date'].max(), periods=12, freq='M').to_period('M').astype(str)
    future_indices = range(len(monthly_sales), len(monthly_sales) + len(future_months))
    future_month_sin = np.sin(2 * np.pi * (np.arange(1, 13)) / 12)
    future_month_cos = np.cos(2 * np.pi * (np.arange(1, 13)) / 12)
    future_data = pd.DataFrame({
        'Month_sin': future_month_sin,
        'Month_cos': future_month_cos,
        'Lag_Sales': [monthly_sales['Sales'].iloc[-1]] * 12
    })
    future_sales = best_model.predict(future_data)
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


if __name__ == '__main__':
    app()
