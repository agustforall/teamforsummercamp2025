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
