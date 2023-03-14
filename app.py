# -*- coding: utf-8 -*-
'''
Created on Mon Oct 25 11:35:11 2021

@author: shangfr
'''
import os
import pickle
import pandas as pd
import streamlit as st
from io import BytesIO
from ui import data_exploration, data_modeling, model_prediction
from charts import e_bar, e_scatter, heatmap, e_roc, e_pr


@st.cache_data(show_spinner=False)
def get_file_content_as_string(path):
    '''get file content as string.
    '''
    with open(path, 'r+', encoding='utf-8') as fo:
        content = fo.read()
    return content


def read_uploaded_file(uploaded_file):
    '''read uploaded file.
    '''
    dtype = uploaded_file.name.split('.')[-1]
    if dtype in ['csv', 'txt']:
        df = pd.read_csv(uploaded_file)
    elif dtype in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)

    dts = df.dtypes
    var_count = df.nunique(axis=0, dropna=True)

    df_dt = pd.DataFrame(
        {'dtypes': dts.values.astype(str)}, index=dts.index)
    df_dt = df_dt.join(var_count.rename('var_count'))
    df_dt['effective'] = [True]*len(dts)
    df_dt.index.name = 'variable'
    df_dt.reset_index(inplace=True)

    origin = {'data': df, 'dtype_table': df_dt}

    return origin


def init_state():
    '''cache_data session_state Initialization.
    '''
    if 'cache_data' not in st.session_state:
        st.session_state['cache_data'] = {'origin': {},
                                          'parm_data': {},
                                          'datasets': {},
                                          'parm_ml': {'ml_type': '有监督',
                                                      'tgt_type': '分类',
                                                      'cls_n': 2},
                                          'parm_model': {'score_criterion': ''},
                                          'output_pipe': {'preprocessor': '',
                                                          'model': '',
                                                          'report': {}}}

    if 'disabled' not in st.session_state:
        st.session_state['disabled'] = False

    if 'file_id' not in st.session_state:
        st.session_state['file_id'] = 0

    if 'ml_step' not in st.session_state:
        st.session_state['ml_step'] = 0


def ml_step_0():
    '''delete session_state.
    '''
    for key in st.session_state.keys():
        del st.session_state[key]

    st.sidebar.warning('请先上传数据集', icon='👆')

    init_state()

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


def pickle_model(model):
    '''Pickle the model inside bytes.
    '''
    f = BytesIO()
    pickle.dump(model, f)
    return f


def show_download(cache_data):
    '''show download for preprocessing data and trained model.
    '''
    col0, col1, col2 = st.sidebar.columns([1, 1, 1])

    X = cache_data['datasets']['X']
    sk_model = cache_data['output_pipe']['model']

    col0.download_button(
        label='📝',
        data=pd.DataFrame(X).to_csv(index=False).encode('utf-8'),
        file_name='preprocessing_df.csv',
        mime='text/csv',
        help='download the preprocessing dataframe.'
    )

    col2.download_button(
        label='💠',
        data=pickle_model(sk_model),
        file_name='model.pkl',
        help='download the trained model.'
    )


@st.cache_resource
def load_pickle(file_name='dict_file.pkl'):
    '''load pickle.
    '''
    with open(file_name, 'rb') as fr:
        cache_data = pickle.load(fr)
    return cache_data


def cache_save(file_name='dict_file.pkl'):
    '''save cache_data.
    '''
    with open(file_name, 'wb') as f_save:
        pickle.dump(st.session_state['cache_data'], f_save)


def model_training():
    '''model training.
    '''
    init_state()
    cache_data = st.session_state['cache_data']
    uploaded_file = st.sidebar.file_uploader('上传数据', type=['xlsx', 'csv'])

    if uploaded_file is None:
        ml_step_0()
        st.stop()

    file_id = uploaded_file.id

    if file_id != st.session_state['file_id']:
        cache_data['origin'] = read_uploaded_file(uploaded_file)
        st.session_state['file_id'] = file_id
        st.session_state['ml_step'] = 1
        st.sidebar.success('数据更换成功', icon="✅")
    else:
        st.sidebar.success('数据上传成功', icon="✅")

    if st.session_state['ml_step'] >= 1:
        data_exploration(cache_data)

        if st.session_state['ml_step'] == 1:
            st.warning('请点击🔧进行数据预处理', icon='⚠️')
        else:
            st.sidebar.success('已完成数据预处理', icon="📝")
            # st.json(cache_data['parm_ml'])

            data_modeling(cache_data)
            if st.session_state['ml_step'] == 2:
                st.warning('请点击🔧进行模型训练', icon='⚠️')

            if st.session_state['ml_step'] >= 3:
                with st.sidebar:
                    st.success('已完成模型训练', icon="💠")
                    agree = st.checkbox('Save Modle')
                    if agree:
                        title = st.text_input('Modle Name', '')
                        if title:
                            cache_save(f'tmp/{title}.pkl')
                            st.info('Model saved successfully.')

                st.text(cache_data['output_pipe']['report']['score'])

                if cache_data['fig_data'].get('feature_importance'):
                    result = cache_data['fig_data']['feature_importance']
                    e_bar(result)
                if cache_data['fig_data'].get('cm'):
                    result = cache_data['fig_data']['cm']
                    heatmap(result)

                if cache_data['fig_data'].get('roc'):
                    fig1, fig2 = st.columns([1, 1])
                    with fig1:
                        e_roc(cache_data['fig_data']['roc'])
                    with fig2:
                        e_pr(cache_data['fig_data']['pr'])

                if cache_data['fig_data'].get('cluster'):
                    result = cache_data['fig_data']['cluster']
                    e_scatter(result)


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
                                    "---", "Training", "Prediction"])

    if app_mode == "---":
        st.sidebar.success('Choose a task')
        st.markdown(get_file_content_as_string('instructions.md'))
    elif app_mode == "Training":
        st.header('Clustering, Classification and Regression')
        model_training()
    elif app_mode == "Prediction":

        st.header('Prediction')
        files = os.listdir('tmp')
        files_opt = st.sidebar.selectbox('模型选择:', files)
        if files_opt:
            cache_data = load_pickle('tmp/' + files_opt)
            show_download(cache_data)
            model_prediction(cache_data)
        else:
            st.warning('Model Training First!', icon="⚠️")

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
