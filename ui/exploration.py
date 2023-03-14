# -*- coding: utf-8 -*-
'''
Created on Mon Oct 25 11:35:11 2021

@author: shangfr
'''
import json
import streamlit as st
from model.preprocessing import transformer


def del_p():
    '''delete positive.
    '''
    st.session_state['cache_data']['parm_ml']['positive'] = ''


def del_t():
    '''delete target and score criterion.
    '''
    st.session_state['cache_data']['parm_ml']['target'] = ''
    st.session_state['cache_data']['parm_model']['score_criterion'] = ''
    del_p()


def show_data(origin):
    '''EDA.
    '''
    data = origin['data']
    dtype_table = origin['dtype_table']

    st.info('1. 数据描述(EDA)', icon='👇')
    tab1, tab2 = st.tabs(['变量描述', '查看数据'])

    with tab1:
        (row_n, col_n) = data.shape

        col1, col2 = st.columns([1, 9])
        col1.metric(label='Data Shape', value=str(col_n)+'列',
                          delta=str(row_n)+'行', delta_color='inverse')
        edited_df = col2.experimental_data_editor(dtype_table)

    with tab2:
        st.dataframe(data)

    filter0 = edited_df['effective'] == True

    return edited_df[filter0]


def data_exploration(cache_data):
    '''ML type select.
    '''
    origin = cache_data['origin']
    data = origin['data']

    dtype_table = show_data(origin)

    st.info('2. 学习类型(ML type)', icon='👇')

    variable = dtype_table['variable']

    filter1 = dtype_table['dtypes'] != 'object'
    num_cols = variable[filter1]

    parm_ml = cache_data['parm_ml']
    hash_value = hash(json.dumps(parm_ml))

    if 'ml_type' not in st.session_state:
        st.session_state.ml_type = parm_ml['ml_type']

    ml_tlist = ['有监督', '无监督']
    t_tlist = ['分类', '回归']

    col1, col2, col3 = st.columns(3)
    ml_type = col1.selectbox('学习类型', ml_tlist, key='ml_type')

    parm_ml['ml_type'] = ml_type

    if ml_type == '有监督':
        if parm_ml.get('tgt_type') and 'tgt_type' not in st.session_state:
            st.session_state.tgt_type = parm_ml['tgt_type']

        tgt_type = col2.selectbox(
            '目标类型', t_tlist, key='tgt_type', on_change=del_t)
        parm_ml['tgt_type'] = tgt_type

        if tgt_type == '分类':
            if parm_ml.get('cls_n') and 'cls_n' not in st.session_state:
                st.session_state.cls_n = parm_ml['cls_n']
            cls_n = col2.number_input(
                '类别数', 2, 3, key='cls_n', help='二分类或多分类', on_change=del_t)
            parm_ml['cls_n'] = cls_n
            filter2 = dtype_table['var_count'] == cls_n

            tar_var_cls = variable[filter2].tolist()

            if len(tar_var_cls) > 0:
                if parm_ml.get('target') and 'target' not in st.session_state:
                    st.session_state.target = parm_ml['target']

                target = col3.selectbox(
                    '目标变量', tar_var_cls, key='target', help='二分类或多分类，且类别数<5', on_change=del_p)
            else:
                col1.error(f'{cls_n}分类模型目标变量不存在！', icon='🚨')
                st.stop()

            t_list = data[target].unique().tolist()
            if parm_ml.get('positive') and 'positive' not in st.session_state:
                st.session_state.positive = parm_ml['positive']

            positive = col3.selectbox(
                'True Positive', t_list, key='positive', help='Value of positive class')
            p_per_t = (data[target] == positive).value_counts(normalize=True)
            if p_per_t[True] < 0.25:
                st.error(
                    f'{cls_n}分类模型目标变量样本不均衡，{positive}占比{p_per_t[True]}小于0.25。', icon='🚨')
                st.stop()
            parm_ml['positive'] = positive
        else:

            if len(num_cols) > 0:
                if parm_ml.get('target') and 'target_n' not in st.session_state:
                    st.session_state.target_n = parm_ml['target']
                target = col3.selectbox(
                    '目标变量', num_cols, key='target_n', help='只能是连续型数值变量', on_change=del_p)
            else:

                col3.error('回归模型目标变量不存在！', icon='🚨')
                st.stop()
            #data[target] = pd.to_numeric(data[target], errors='coerce')
        parm_ml['target'] = target
        variable = variable[variable != target].tolist()
        num_cols = num_cols[num_cols != target].tolist()

    elif ml_type == '无监督':
        pass

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

    parm_ml['features'] = features
    parm_ml['max_n'] = max_n
    if hash_value != hash(json.dumps(parm_ml)):
        st.session_state['ml_step'] = 1

    cache_data['parm_ml'] = parm_ml
    submitted = st.button('🔧 预处理')
    if submitted:

        datasets, preprocessor = transformer(cache_data)

        cache_data['datasets'] = datasets
        cache_data['output_pipe']['preprocessor'] = preprocessor
        st.session_state['ml_step'] = 2
