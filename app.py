# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:35:11 2021

@author: shangfr
"""

import streamlit as st

st.set_page_config(
    page_title="Data Follower",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# A Machine Learning Web App with Streamlit and Python"
    })


def run():
    st.sidebar.title("An Interactive ML WebApp")
    st.sidebar.header("Clustering, Classification and Regression")

    @st.cache(show_spinner=False)
    def get_file_content_as_string(path):
        fo = open(path, "r+", encoding='utf-8')
        return fo.read()

    readme_text = st.markdown(get_file_content_as_string("README.md"))

    #app_mode = st.sidebar.selectbox("Select a task above.", ["---", "Clustering", "Classification", "Regression"])
    app_mode = st.sidebar.selectbox("Select a task above.", ["---", "Model Training", "Model Prediction"])

    if app_mode == "---":
        st.sidebar.success('Choose a task')
    elif app_mode == "Model Training":
        from ui import ui_training
        readme_text.empty()
        ui_training()
    elif app_mode == "Model Prediction":
        from ui import ui_prediction
        readme_text.empty()
        ui_prediction()



if __name__ == "__main__":
    run()
