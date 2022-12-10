import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def app():        

    st.title("Software Developer Salary Prediction")
    df = pd.read_csv('survey.csv')
    dataset = pd.read_csv('survey_no.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    X = X[:, 1:]

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    """from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

    # Fitting Multiple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    st.bar_chart(y_pred)
    df = df.head(15)
    st.table(df.style.hide())
    st.table(y_pred)
    
    