# -*- coding: utf-8 -*-
'''
Created on Mon Oct 25 11:35:11 2021

@author: shangfr
'''
import pandas as pd
import streamlit as st
from model.preprocessing import transformer


def reset_step():
    '''reset_step.
    '''
    st.session_state['ml_step'] = 1


def show_data(origin):
    '''EDA.
    '''
    data = origin['data']
    dtype_table = origin['dtype_table']

    st.info('1. 数据描述(EDA)', icon='👇')
    tab1, tab2 = st.tabs(['变量描述', '查看数据'])

    with tab1:
        (row_n, col_n) = data.shape

        col1, col2 = st.columns([2, 8])
        col1.metric(label='Data Shape', value=str(col_n)+'列',
                          delta=str(row_n)+'行', delta_color='inverse')
        edited_df = col2.experimental_data_editor(dtype_table)

    with tab2:
        st.dataframe(data.head(200))

    filter0 = edited_df['effective'] == True

    return edited_df[filter0]


@st.cache_data
def convert_df(X, feature_names):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = pd.DataFrame(X, columns=feature_names)
    return df.to_csv(index=False).encode('utf-8')


def data_exploration(cache_data):
    '''ML type select.
    '''
    origin = cache_data['origin']
    data = origin['data']

    dtype_table = show_data(origin)

    st.info('2. 机器学习(Machine Learning)', icon='👇')

    variable = dtype_table['variable']

    filter1 = dtype_table['dtypes'] != 'category'
    num_cols = variable[filter1]

    machine_learning = cache_data['machine_learning']
    ml_parm = machine_learning['parm']

    m_type_lst = ['分类', '回归', '聚类']
    col1, col2, col3 = st.columns(3)
    model_type = col1.selectbox(
        '类型', m_type_lst, key='m_type', on_change=reset_step)
    ml_parm['model_type'] = model_type
    if model_type == m_type_lst[0]:
        cls_n = col2.number_input(
            '类别数', 2, 3, key='cls_n', help='二分类或多分类')
        ml_parm['cls_n'] = cls_n
        filter2 = dtype_table['var_count'] == cls_n

        tar_var_cls = variable[filter2].tolist()

        if len(tar_var_cls) > 0:
            target = col3.selectbox(
                '目标变量', tar_var_cls, key='target', help='二分类或多分类，且类别数<5')

        else:
            col1.error(f'{cls_n}分类模型目标变量不存在！', icon='🚨')
            st.stop()

        t_list = data[target].unique().tolist()

        positive = col3.selectbox(
            'True Positive', t_list, key='positive', help='Value of positive class')
        ml_parm['positive'] = positive
        p_per_t = (data[target] == positive).value_counts(normalize=True)
        if p_per_t[True] < 0.25:
            st.error(
                f'{cls_n}分类模型目标变量样本不均衡，{positive}占比{p_per_t[True]}小于0.25。', icon='🚨')
            st.stop()

    elif model_type == m_type_lst[1]:
        if len(num_cols) > 0:
            filter3 = dtype_table['var_count'] > 10

            tar_var_regr = variable[filter3].tolist()

            target = col3.selectbox(
                '目标变量', tar_var_regr, key='target_n', help='只能是连续型数值变量')

        else:
            col3.error('回归模型目标变量不存在！', icon='🚨')
            st.stop()
        #data[target] = pd.to_numeric(data[target], errors='coerce')

    elif model_type == m_type_lst[2]:
        target = ''

    variable = variable[variable != target].tolist()
    num_cols = num_cols[num_cols != target].tolist()

    dtypes = ['数值型', '分类型', '时间型', '描述型']
    variable = list(variable)
    features_num_cols = st.multiselect(
        f'特征变量({dtypes[0]})', list(num_cols), list(num_cols), disabled=st.session_state.disabled)
    variable = [c for c in variable if c not in features_num_cols]

    features_cat_cols = st.multiselect(
        f'特征变量({dtypes[1]})', list(variable), list(variable), disabled=st.session_state.disabled)
    variable = [c for c in variable if c not in features_cat_cols]

    features = {'num_cols': features_num_cols, 'cat_cols': features_cat_cols}

    max_n = len(features_num_cols) + len(features_cat_cols)
    if max_n < 2:
        st.error('Number of features cannot be less than 2.', icon='🚨')
        st.stop()

    ml_parm['target'] = target
    ml_parm['features'] = features
    ml_parm['feature_names'] = features['num_cols']+features['cat_cols']
    ml_parm['max_n'] = max_n

    submitted = st.button('🔧 预处理')
    if submitted:

        datasets, preprocessor = transformer(data, ml_parm)

        machine_learning['datasets'] = datasets
        machine_learning['model_pipe']['preprocessor'] = preprocessor
        st.session_state['ml_step'] = 2

    if st.session_state['ml_step'] == 1:
        st.warning('请先点击🔧进行数据预处理', icon='⚠️')
        st.stop()

    col0, col1 = st.sidebar.columns([1, 5])
    col1.success('已完成数据预处理')

    X = machine_learning['datasets']['X']
    feature_names = machine_learning['parm']['feature_names']
    csv_data = convert_df(X, feature_names)
    col0.download_button(
        label='📝',
        data=csv_data,
        file_name='preprocessing_df.csv',
        mime='text/csv',
        help='download the preprocessing dataframe.'
    )
