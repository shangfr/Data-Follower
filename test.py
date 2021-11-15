# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:21:25 2021

@author: shangzhu
"""

from datetime import datetime, timedelta
from sqlalchemy import create_engine
import pandas as pd
import tushare as ts

def make_pickle(con):
    
    sql_cmd = "SELECT ts_code,trade_date,open,high,low,close,pct_chg FROM daily_data;"
    df = pd.read_sql(sql=sql_cmd, con=con)
    #df_daily = pd.read_sql_table("daily_data", con)
    #df = df_daily[['ts_code', 'trade_date', 'open','high', 'low', 'close', 'pct_chg']].copy()
    df.drop_duplicates(subset=['ts_code', 'trade_date'],
                       keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    matrix_close = df.pivot(
        index='trade_date', columns='ts_code', values='close')
    matrix_close.fillna(0, inplace=True)
    matrix_close.index = pd.DatetimeIndex(matrix_close.index).date

    matrix_open = df.pivot(
        index='trade_date', columns='ts_code', values='open')
    matrix_open.fillna(0, inplace=True)
    matrix_open.index = pd.DatetimeIndex(matrix_open.index).date

    matrix_high = df.pivot(
        index='trade_date', columns='ts_code', values='high')
    matrix_high.fillna(0, inplace=True)
    matrix_high.index = pd.DatetimeIndex(matrix_high.index).date

    matrix_low = df.pivot(index='trade_date', columns='ts_code', values='low')
    matrix_low.fillna(0, inplace=True)
    matrix_low.index = pd.DatetimeIndex(matrix_low.index).date

    matrix_chg = df.pivot(index='trade_date',
                          columns='ts_code', values='pct_chg')
    matrix_chg.fillna(0, inplace=True)
    matrix_chg.index = pd.DatetimeIndex(matrix_chg.index).date

    matrix_lim = matrix_chg > 9.9
    matrixlim = matrix_lim.rolling(20).sum()

    matrix10 = matrix_close.rolling(10).mean()
    matrix20 = matrix_close.rolling(20).mean()
    matrix30 = matrix_close.rolling(30).mean()

    ts20 = (matrixlim > 0)
    # 20日
    ts20.to_pickle('model/out_put1.pickle')
    ts1020 = (matrix10 > matrix20)
    ts2030 = (matrix20 > matrix30)

    ts123 = ts1020 & ts2030
    # 上升
    ts123.to_pickle('model/out_put2.pickle')

    # 金针
    tsol = (matrix_open*0.98 > matrix_low)
    tsol.to_pickle('model/out_put3.pickle')
    # 探底
    tsl20 = (matrix_low < matrix20*1.005) & (matrix_low > matrix20*0.995)
    tsl20.to_pickle('model/out_put4.pickle')
    # 光头

    tsch = (1.005*matrix_close >= matrix_high)
    tschg = matrix_chg > 0
    out_put = tsch & tschg
    out_put.to_pickle('model/out_put5.pickle')

    # 中阳
    tsoc = (matrix_close >= matrix_open*1.03)

    tsoc.to_pickle('model/out_put6.pickle')

    # 小阳
    tsxy = (matrix_open*1.03 >= matrix_close) & (matrix_close >= matrix_open*1.005)

    tsxy.to_pickle('model/out_put7.pickle')



token = '1b192ba1b6f325e91f14ae888c25165a090f6470a50639d49656e648'
pro = ts.pro_api(token)

curr_date = datetime.today().date()
trade_date = curr_date.strftime("%Y%m%d")

df = pro.daily(trade_date=trade_date)
table_name = 'daily_data'


file = 'sqlite:///'
db_name = 'stock_data.db'
engine = create_engine(file+db_name)
df.to_sql(table_name, engine, index=False, if_exists='append')
print("已更新当日数据")
#make_pickle(engine)
#print("已更新当日形态")