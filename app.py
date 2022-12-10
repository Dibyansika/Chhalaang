import streamlit as st
from multiapp import MultiApp
from apps import app1,app2 # import your app modules here
app = MultiApp()

st.markdown("""
# Salary and Loan Prediction
""")

# page = st.sidebar.selectbox("Salary Prediction and Explore", ("Predict", "Explore"))
# Add all your application here
app.add_app("Predict", app1.app)
app.add_app("Explore", app2.app)
# The main app
app.run()

