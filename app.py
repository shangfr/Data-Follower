# -*- coding: utf-8 -*-
'''
Created on Mon Oct 25 11:35:11 2021

@author: shangfr
'''
import pandas as pd
import streamlit as st
from ui import data_exploration, data_analysis
from ui import data_modeling, result_display
from ui import model_prediction
from utils import describe, load_pickle


@st.cache_data(show_spinner=False)
def get_file_content_as_string(path):
    '''get file content as string.
    '''
    with open(path, 'r+', encoding='utf-8') as fo:
        content = fo.read()
    return content


def init_state():
    '''cache_data session_state Initialization.
    '''
    if 'cache_data' not in st.session_state:
        st.session_state['cache_data'] = {
            'origin': {},
            'machine_learning': {
                'parm': {},
                'model_pipe': {}
            },
            'output': {}
        }

    if 'loaded' not in st.session_state:
        st.session_state['loaded'] = False

    if 'disabled' not in st.session_state:
        st.session_state['disabled'] = False

    if 'file_id' not in st.session_state:
        st.session_state['file_id'] = 0

    if 'ml_step' not in st.session_state:
        st.session_state['ml_step'] = 0


def reset_state():
    '''delete session_state and init.
    '''
    for key in st.session_state.keys():
        del st.session_state[key]
    init_state()


def load_state():

    cache_data = load_pickle()

    reset_state()
    st.session_state['cache_data'] = cache_data
    st.session_state['ml_step'] = 3
    st.session_state['loaded'] = True
    st.session_state['disabled'] = True


def show_ml_step():
    '''show ml step.
    '''
    st.header('1. Methods of data exploration')
    ui_info1 = '''
                `EDA`
                > Data exploration, also known as exploratory data analysis (EDA), is a process where users look at and understand their data with statistical and visualization methods. 
                
                > 数据探索EDA，即探索性数据分析，是用户使用统计和可视化方法查看和理解数据的过程。
            '''

    st.markdown(ui_info1)

    st.header('2. Train a machine learning model')
    ui_info2 = '''
                `Clustering, Classification and Regression`
                > Machine learning is a branch of artificial intelligence. It focuses on using data and algorithms to imitate human learning methods and gradually improve the accuracy of simulation.
                
                > 机器学习是人工智能的一个分支，其重点是使用数据和算法模仿人类的学习方式，并逐步提高模仿的准确性。
            '''

    st.markdown(ui_info2)
    st.stop()


def read_uploaded_file():
    '''read uploaded file.
    '''
    uploaded_file = st.sidebar.file_uploader(
        '上传数据', type=['xlsx', 'csv'], key='dfa')
    if uploaded_file is None:
        st.sidebar.warning('请先上传数据集', icon='👆')
        reset_state()
        show_ml_step()
    file_id = uploaded_file.id
    if file_id != st.session_state['file_id']:
        dtype = uploaded_file.name.split('.')[-1]
        if dtype in ['csv', 'txt']:
            df = pd.read_csv(uploaded_file)
        elif dtype in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)

        st.session_state['cache_data']['origin'] = describe(df)
        st.session_state['file_id'] = file_id
        st.session_state['ml_step'] = 1
        st.sidebar.success('数据更换成功', icon="✅")
    else:
        st.sidebar.success('数据上传成功', icon="✅")

    return st.session_state['cache_data']


def model_training():
    '''model training.
    '''

    init_state()

    # 读取或更新数据
    cache_data = read_uploaded_file()
    # 数据探索与预处理
    data_exploration(cache_data)
    # 数据分析
    st.info('3. 模型特征分析(Features Analysis)', icon='👇')
    data_analysis(cache_data)
    # 数据建模
    data_modeling(cache_data)
    result_display(cache_data)


if __name__ == '__main__':

    st.set_page_config(
        page_title='Data Follower',
        page_icon='🦄',
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            'About': '# A Machine Learning Web App with Streamlit and Python'
        })

    st.sidebar.title('An Interactive ML WebApp')

    app_mode = st.sidebar.selectbox("Select a task above.", [
                                    "---", "Training", "Application"])
    # Style
    with open('style.css')as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if app_mode == "---":
        st.sidebar.success('Choose a task')
        st.markdown(get_file_content_as_string('instructions.md'))
    elif app_mode == "Training":
        st.header('Clustering, Classification and Regression')
        model_training()
    elif app_mode == "Application":

        load_state()
        st.sidebar.success("Model loaded successfully. ", icon="👆")
        tool_mode = st.sidebar.selectbox("Model", [
            "Checking", "Prediction"])
        if tool_mode == "Prediction":
            model_prediction(st.session_state['cache_data'])
        elif tool_mode == "Checking":
            st.info('1. 模型特征分析(Features Analysis)', icon='👇')
            data_analysis(st.session_state['cache_data'])
            st.info('2. 模型展示(Model Display)', icon='👇')
            result_display(st.session_state['cache_data'])
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/kuake2022">@kuake2022</a></h6>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin-top: 0.75em;"><a href="https://www.buymeacoffee.com/kuake" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
            unsafe_allow_html=True,
        )
