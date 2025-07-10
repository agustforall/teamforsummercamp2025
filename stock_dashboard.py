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

# 股票搜索功能
@st.cache_data
def search_stock(keyword):
    """根据关键字搜索股票"""
    if keyword.isdigit():
        # 按代码搜索，直接获取所有股票并筛选
        try:
            all_stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,area,industry,list_date')
            if all_stocks.empty:
                st.error("无法获取股票列表，请检查API密钥权限或网络连接")
                return pd.DataFrame()
            # 提取纯数字代码进行匹配
            all_stocks['pure_code'] = all_stocks['ts_code'].str.split('.').str[0]
            # 精确匹配代码
            df = all_stocks[all_stocks['pure_code'] == keyword]
            # 如果精确匹配不到，尝试模糊匹配
            if df.empty:
                df = all_stocks[all_stocks['pure_code'].str.contains(keyword)]
            return df
        except Exception as e:
            st.error(f"获取股票数据失败: {str(e)}")
            return pd.DataFrame()
    else:
        # 按名称搜索，先精确匹配
        df = pro.stock_basic(exchange='', list_status='L', name=keyword, fields='ts_code,name,area,industry,list_date')
        # 如果精确匹配为空，尝试模糊匹配
        if df.empty:
            all_stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,area,industry,list_date')
            df = all_stocks[all_stocks['name'].str.contains(keyword, case=False)]
    return df
    
# 获取股票数据
@st.cache_data
def get_stock_data(ts_code, start_date, end_date):
    """获取股票日线数据"""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    df = pro.daily(ts_code=ts_code, start_date=start_str, end_date=end_str)
    if df.empty:
        return None
    # 转换日期格式
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    # 按日期排序
    df = df.sort_values('trade_date')
    # 设置日期为索引
    df.set_index('trade_date', inplace=True)
    # 计算技术指标
    calculate_technical_indicators(df)
    return df

# 计算技术指标
def calculate_technical_indicators(df):
    """计算各种技术指标"""
    # MACD
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(
        df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # RSI
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
    
    # KDJ
    df['k'], df['d'] = talib.STOCH(
        df['high'].values, df['low'].values, df['close'].values,
        fastk_period=9, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0)
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    # 均线
    df['ma5'] = talib.SMA(df['close'].values, timeperiod=5)
    df['ma10'] = talib.SMA(df['close'].values, timeperiod=10)
    df['ma20'] = talib.SMA(df['close'].values, timeperiod=20)
    df['ma60'] = talib.SMA(df['close'].values, timeperiod=60)
    
    # 布林带
    df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(
        df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    
    return df

# 训练简单的AI模型
def train_ai_model(df):
    """基于技术指标训练简单的分类模型"""
    # 创建目标变量：明天收盘价是否高于今天
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    # 选择特征
    features = ['macd', 'macdsignal', 'macdhist', 'rsi', 'k', 'd', 'j', 'ma5', 'ma10', 'ma20', 'upperband', 'middleband', 'lowerband']
    
    # 删除含有NaN值的行
    df = df.dropna(subset=features + ['target'])
    
    if len(df) < 30:
        return None, "样本量不足，无法训练模型"
    
    X = df[features]
    y = df['target']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
# 预测并计算准确率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, f"模型准确率: {accuracy:.2f}"

# 生成AI投资建议
def generate_ai_recommendation(model, latest_data):
    """基于最新数据生成投资建议"""
    if model is None:
        return "无法生成建议: 模型未训练"
    
    features = ['macd', 'macdsignal', 'macdhist', 'rsi', 'k', 'd', 'j', 'ma5', 'ma10', 'ma20', 'upperband', 'middleband', 'lowerband']
    
    # 检查是否有足够的数据
    if latest_data is None or len(latest_data) < 1:
        return "无法生成建议: 缺少数据"
    
    # 获取最新的特征数据
    try:
        latest_features = latest_data[features].iloc[-1:]
    except KeyError as e:
        return f"无法生成建议: 缺少特征 {e}"
    
    # 预测
    prediction = model.predict(latest_features)
    probability = model.predict_proba(latest_features)[0][prediction[0]]
    
    # 生成建议
    if prediction[0] == 1:
        return f"📈 AI建议: 买入 (概率: {probability:.2f})\n基于技术指标分析，预计股价将上涨"
    else:
        return f"📉 AI建议: 卖出 (概率: {probability:.2f})\n基于技术指标分析，预计股价将下跌"
