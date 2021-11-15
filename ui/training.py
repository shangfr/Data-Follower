# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:35:11 2021

@author: shangfr
"""

import pandas as pd
import streamlit as st

from model import model_cls, model_regr, model_cluster

st.header("Train a machine learning model")
ui_info = """
            `Clustering, Classification and Regression`
            > 机器学习是人工智能的一个分支，其重点是使用数据和算法模仿人类的学习方式，并逐步提高模仿的准确性。
        """

st.markdown(ui_info)


def data_drop(data,filter_dict):

    data.dropna(axis=1, how='all', inplace=True)
    data.drop_duplicates(subset=None, keep='first', inplace=True)
    return data


def data_setup(col1,col2,data):
    f_drop = col2.checkbox('数据去重', value=True, help='将完全重复的行数据去除')
    f_encoder = col2.checkbox('特征编码', value=True, help='将字符型特征转换为数值型')

    na_rate = col1.number_input(
        '缺失值处理', 10, 100, 95, help='设定缺失值比例，删除大于该比例的特征')
    pad_type = ''
    if na_rate < 20:
        f_pad = col1.checkbox('填充', help='缺失值比例较少，可以对缺失数据进行填充')
        if f_pad:
            pad_type = col1.selectbox('填充方式', ['均值', '顺序'])

    filter_dict = {"f_drop": f_drop, "f_encoder": f_encoder,
                   "na_rate": na_rate, "pad_type": pad_type}


    col1, col2 = st.columns(2)
    if st.button('🔧'):
        with st.spinner('Wait for it...'):
            df = data_drop(data, filter_dict)

        (row_n, col_n) = df.shape
        col1.metric(label="Cleaning Data Shape", value=str(
            col_n)+'列', delta=str(row_n)+'行', delta_color="inverse")
        col2.success('已完成数据预处理，点击下载数据。')
        col2.download_button(
            label="📝",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='preprocessing_df.csv',
            mime='text/csv',
        )
        
        return df
    
    else:
        
        return pd.DataFrame()
    
def model_setup(data, variable,tar_var_cls,tar_var_regr):
    col1, col2, col3 = st.columns(3)
    col21, col22 = st.columns([2, 1])
    target, positive, t_type = '', '', ''

    ml_type = col1.selectbox('学习类型', ['有监督', '无监督'])
    if ml_type == '有监督':
        t_type = col2.selectbox('目标类型', ['分类', '回归'])
        if t_type == '分类':
            target = col3.selectbox('目标变量', tar_var_cls)
            positive = col22.selectbox('True Positive', data[target].unique(
            ), help='Value of positive class')
        else:
            target = col3.selectbox('目标变量', tar_var_regr)
        variable = variable.drop(target)

    variable = variable.tolist()
    features = col21.multiselect('特征变量', variable, variable)

    model_dict = {"data":data, "ml_type": ml_type, "features": features,
                  "target": target, "positive": positive, "t_type": t_type}
    
    return model_dict


def init_state():
    # Initialization
    if 'file_id' not in st.session_state:
        st.session_state['file_id'] = 0
        
    if 'ml_step' not in st.session_state:
        st.session_state['ml_step'] = 1

    if 'dataset_dict' not in st.session_state:
        st.session_state['dataset_dict'] = {}

    if 'model_dict' not in st.session_state:
        st.session_state['model_dict'] = {}

    if 'filter_dict' not in st.session_state:
        st.session_state['filter_dict'] = {}


def update_state(dataset_dict,file_id=None):

    st.session_state['dataset_dict'] = dataset_dict
    if file_id!=None:
        st.session_state['file_id'] = file_id
        
    print("更新数据集")

def get_dataset_dict(data):
    variable = data.columns
    var_type = data.dtypes.unique()
    var_count = data.nunique(axis=0, dropna=True)
    tar_var_cls = var_count[var_count < 5].index
    tar_var_regr = var_count[var_count > 5].index

    dataset_dict = {"data":data,"variable": variable, "var_type": var_type,
                    "tar_var_cls": tar_var_cls, "tar_var_regr": tar_var_regr}
    return dataset_dict

    
def ui_training():
    init_state()

    with st.expander("1. 数据预处理", expanded=(st.session_state['ml_step'] == 1)):
        uploaded_file = st.file_uploader("Choose a file", type=['xlsx'])
        if uploaded_file is not None:
            if uploaded_file.id != st.session_state['file_id']:
                
                data = pd.read_excel(uploaded_file)
                dataset_dict = get_dataset_dict(data)
                file_id = uploaded_file.id
                update_state(dataset_dict,file_id)
                st.session_state['ml_step'] = 1
                
            else:
                print("读取缓存数据")
                data = st.session_state['dataset_dict']["data"]

            (row_n, col_n) = data.shape
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="Data Shape", value=str(col_n)+'列',
                        delta=str(row_n)+'行', delta_color="inverse")

            show_data = col2.selectbox(
                'Data Info', ['单变量', '查看数据', '统计描述', '预处理'], help='确认后可以进行下一步')

            if show_data == '单变量':
                vtype = col3.selectbox(
                    'Univariate Type', st.session_state['dataset_dict']["var_type"])
                df_sl = data.select_dtypes(include=vtype)
                df_col = col4.selectbox('Univariate View', df_sl.columns)
                st.dataframe(df_sl[df_col].value_counts())
            if show_data == '查看数据':
                st.dataframe(data)

            if show_data == '统计描述':
                # 查看不同数值类型特征的统计描述

                df_num = data.select_dtypes(exclude=['object'])
                df_obj = data.select_dtypes(include=['object'])
                if df_num.size != 0:
                    st.dataframe(df_num.describe())
                if df_obj.size != 0:
                    st.code(df_obj.describe())

            if show_data == '预处理':
                # 统计目标变量类型
                clean_df = data_setup(col3,col4,data)
                if clean_df.empty:
                    pass
                else:
                    dataset_dict = get_dataset_dict(clean_df)
                    update_state(dataset_dict)
                    st.session_state['ml_step'] = 2
                    st.balloons()
      
                if st.session_state['ml_step'] > 1:
                    st.success("数据集已确认")

    with st.expander("2. 模型设置", expanded=(st.session_state['ml_step'] == 2)):
        if st.session_state['ml_step'] >= 2:
            data = st.session_state['dataset_dict']["data"]
            variable = st.session_state['dataset_dict']['variable']
            tar_var_cls = st.session_state['dataset_dict']['tar_var_cls']
            tar_var_regr = st.session_state['dataset_dict']['tar_var_regr']
            model_dict = model_setup(data,variable,tar_var_cls,tar_var_regr)
            st.session_state['model_dict'] = model_dict
            st.session_state['ml_step'] = 3
        else:
            st.warning("请先进行数据预处理")
            
    with st.expander("3. 模型训练", expanded=(st.session_state['ml_step'] == 3)):
        # st.write(st.session_state['ml_step'])
        if st.session_state['ml_step'] == 3:
            model_dict = st.session_state['model_dict']
            if st.button('训练'):
                if model_dict['ml_type'] == '无监督':
                    fig = model_cluster(model_dict)
                    st.plotly_chart(fig, use_container_width=True)
                else:

                    if model_dict['t_type'] == '分类':
                        fig, report = model_cls(model_dict)
                        st.code(report)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        report = model_regr(model_dict)
                        st.code(report)

                #st.session_state['ml_step'] = 4
        else:
            st.warning("请先进行数据预处理")


if __name__ == "__main__":
    ui_training()
