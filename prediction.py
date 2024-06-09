import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import XGBoost
from xgboost import XGBRegressor

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

    df = load_data()

    # Data Cleaning
    df.drop_duplicates(inplace=True)
    df['Sales'] = df['Sales'].astype(str).str.replace(',', '').astype(float)
    df['Profit'] = df['Profit'].astype(str).str.replace(',', '').astype(float)
    df['Discount'] = df['Discount'].astype(str).str.replace(',', '').astype(float)

    df['Sales'].fillna(df['Sales'].median(), inplace=True)
    df['Profit'].fillna(df['Profit'].median(), inplace=True)
    df['Discount'].fillna(df['Discount'].median(), inplace=True)

    # Add new features
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)
    df['Month'] = df['Order Date'].dt.month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Additional feature engineering (example only, add your own features)
    df['Customer Age'] = 2024 - df['Customer Birth Year']
    df['Product Category'] = df['Product Name'].apply(lambda x: x.split(' - ')[0])

    # Encoding categorical variables
    categorical_cols = ['Product Category', 'Customer Gender']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    X = df_encoded.drop(['Sales', 'Order Date', 'Order Month'], axis=1)
    y = df_encoded['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning for XGBoost
    xgb_param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0, 0.1, 0.5, 1]
    }

    xgb_random_search = RandomizedSearchCV(XGBRegressor(random_state=42), param_distributions=xgb_param_dist, n_iter=100, cv=5, scoring='r2', random_state=42, n_jobs=-1)
    xgb_random_search.fit(X_train, y_train)

    best_xgb_model = xgb_random_search.best_estimator_

    # Model evaluation
    y_pred = best_xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"**Mean Squared Error:** {mse}")
    st.write(f"**R2 Score:** {r2}")
    st.write(f"**Mean Absolute Error:** {mae}")

    # Visualization (add your own visualization as needed)
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': best_xgb_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    st.bar_chart(feature_importance.head(10))

    # Additional visualizations or insights
    # ...

# Run the app
if __name__ == '__main__':
    app()
