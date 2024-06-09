import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
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
    
    # Data Cleaning (unchanged)
    
    # Feature Engineering
    df['Region'] = pd.Categorical(df['Region']).codes  # Assuming 'Region' is a categorical feature
    
    # Ensure proper scaling of features (MinMaxScaler in this example)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[['Sales', 'Profit', 'Discount', 'Quantity']] = scaler.fit_transform(df[['Sales', 'Profit', 'Discount', 'Quantity']])
    
    # Aggregation (unchanged)
    
    # Train-Test Split
    X = df[['Month_sin', 'Month_cos', 'Lagged_Sales', 'Discount', 'Profit', 'Quantity', 'Region']].values
    y = df['Sales'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter tuning for Gradient Boosting Regressor
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10, 15]
    }

    random_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_distributions=param_dist, n_iter=50, cv=5, scoring='r2', random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    # Model Evaluation
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(best_model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot Learning Curve
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("R2 Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    st.pyplot(plt)

    # Predictions and Metrics (unchanged)
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"**Mean Squared Error:** {mse}")
    st.write(f"**R2 Score:** {r2}")
    st.write(f"**Mean Absolute Error:** {mae}")

    # Rest of the code (unchanged)
    st.subheader("📈 Gradient Boosting Model Results")
    # ... plotting and visualizations ...

if __name__ == '__main__':
    app()
