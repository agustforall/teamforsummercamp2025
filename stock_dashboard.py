import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import tushare as ts
import os
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

# 设置页面配置
st.set_page_config(
    page_title="股票行情分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态存储自选股
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# 初始化选中的股票代码
if 'selected_stock_code' not in st.session_state:
    st.session_state.selected_stock_code = '600036'

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载环境变量（作为备选方案）
load_dotenv()
api_key = os.getenv('TUSHARE_API_KEY')

# 如果环境变量中没有API密钥，则让用户在界面输入
if not api_key:
    st.sidebar.header("API配置")
    api_key = st.sidebar.text_input("请输入Tushare API密钥", type="password", help="您可以在Tushare官网获取API密钥")
    if not api_key:
        st.error("请输入Tushare API密钥以继续使用应用")
        st.stop()

# 初始化Tushare
ts.set_token(api_key)
pro = ts.pro_api()

# 标题
st.title("📈 股票行情分析平台")

# 侧边栏 - 股票搜索
st.sidebar.header("股票选择")
stock_code = st.sidebar.text_input(
    "输入股票代码或名称", 
    value=st.session_state.selected_stock_code, 
    help="例如: 600036 或 招商银行"
)

# 侧边栏 - 时间范围选择
st.sidebar.header("时间范围")
end_date = datetime.datetime.now()
start_date = st.sidebar.date_input(
    "开始日期",
    value=end_date - datetime.timedelta(days=365),
    max_value=end_date - datetime.timedelta(days=1)
)
end_date = st.sidebar.date_input(
    "结束日期",
    value=end_date,
    max_value=end_date
)

# 侧边栏 - 技术指标选择
st.sidebar.header("技术指标")
show_macd = st.sidebar.checkbox("MACD", value=True)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_kdj = st.sidebar.checkbox("KDJ", value=True)
show_ma = st.sidebar.checkbox("均线", value=True)
show_boll = st.sidebar.checkbox("布林带", value=True)
show_volume = st.sidebar.checkbox("成交量", value=True)

# 侧边栏 - 自选股
st.sidebar.header("自选股")
if st.session_state.watchlist:
    for i, stock in enumerate(st.session_state.watchlist):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            if st.button(f"{stock['name']} ({stock['ts_code']})", key=f'watch_{i}'):
                st.session_state.selected_stock_code = stock['ts_code'].split('.')[0]
                st.experimental_rerun()
        with col2:
            if st.button('🗑️', key=f'del_{i}', help='删除自选股'):
                st.session_state.watchlist.pop(i)
                st.rerun()
else:
    st.sidebar.info('暂无自选股，搜索股票后可添加')

# 侧边栏 - AI分析设置
st.sidebar.header("AI分析")
use_ai = st.sidebar.checkbox("启用AI投资建议", value=True)
