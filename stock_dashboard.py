import streamlit as st  # Streamlit for building the web UI
from dotenv import load_dotenv  # For loading environment variables from .env file
import pandas as pd  # Data handling
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Visualization
import talib  # Technical analysis indicators
import tushare as ts  # Tushare API for stock market data
import os  # For OS operations like reading environment variables
from sklearn.linear_model import LogisticRegression  # Machine learning model
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.metrics import accuracy_score  # Model evaluation
import datetime  # Handling date and time

# Set Streamlit page configuration (title, icon, layout, etc.)
st.set_page_config(
    page_title="股票行情分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for watchlist if not present
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# Initialize selected stock code in session state
if 'selected_stock_code' not in st.session_state:
    st.session_state.selected_stock_code = '600036'

# Set Chinese font and fix negative sign display issues
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# Load Tushare API key from environment variables
load_dotenv()
api_key = os.getenv('TUSHARE_API_KEY')

# If API key is not set, prompt the user to input it manually
if not api_key:
    st.sidebar.header("API配置")
    api_key = st.sidebar.text_input("请输入Tushare API密钥", type="password", help="您可以在Tushare官网获取API密钥")
    if not api_key:
        st.error("请输入Tushare API密钥以继续使用应用")
        st.stop()

# Initialize Tushare with the API token
ts.set_token(api_key)
pro = ts.pro_api()

# Title of the app
st.title("📈 股票行情分析平台")

# Sidebar - Stock selection input
st.sidebar.header("股票选择")
stock_code = st.sidebar.text_input(
    "输入股票代码或名称", 
    value=st.session_state.selected_stock_code, 
    help="例如: 600036 或 招商银行"
)

# Sidebar - Date range inputs
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

# Sidebar - Technical indicators selection
st.sidebar.header("技术指标")
show_macd = st.sidebar.checkbox("MACD", value=True)  # Moving Average Convergence Divergence
show_rsi = st.sidebar.checkbox("RSI", value=True)  # Relative Strength Index
show_kdj = st.sidebar.checkbox("KDJ", value=True)  # Stochastic Oscillator
show_ma = st.sidebar.checkbox("均线", value=True)  # Moving Averages
show_boll = st.sidebar.checkbox("布林带", value=True)  # Bollinger Bands
show_volume = st.sidebar.checkbox("成交量", value=True)  # Trading Volume

# Sidebar - Watchlist management
st.sidebar.header("自选股")
if st.session_state.watchlist:
    for i, stock in enumerate(st.session_state.watchlist):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            if st.button(f"{stock['name']} ({stock['ts_code']})", key=f'watch_{i}'):
                st.session_state.selected_stock_code = stock['ts_code'].split('.')[0]
                st.experimental_rerun()  # Rerun to update selected stock
        with col2:
            if st.button('🗑️', key=f'del_{i}', help='删除自选股'):
                st.session_state.watchlist.pop(i)  # Remove from watchlist
                st.rerun()
else:
    st.sidebar.info('暂无自选股，搜索股票后可添加')

# Sidebar - AI analysis toggle
st.sidebar.header("AI分析")
use_ai = st.sidebar.checkbox("启用AI投资建议", value=True)  # Whether to use AI prediction

# Search stock by code or name
@st.cache_data
def search_stock(keyword):
    """Search stock info by keyword (code or name)"""
    if keyword.isdigit():  # If input is numeric, treat as stock code
        try:
            all_stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,area,industry,list_date')
            if all_stocks.empty:
                st.error("无法获取股票列表，请检查API密钥权限或网络连接")
                return pd.DataFrame()
            all_stocks['pure_code'] = all_stocks['ts_code'].str.split('.').str[0]  # Extract numeric part of code
            df = all_stocks[all_stocks['pure_code'] == keyword]
            if df.empty:
                df = all_stocks[all_stocks['pure_code'].str.contains(keyword)]  # Fuzzy match
            return df
        except Exception as e:
            st.error(f"获取股票数据失败: {str(e)}")
            return pd.DataFrame()
    else:  # If input is name
        df = pro.stock_basic(exchange='', list_status='L', name=keyword, fields='ts_code,name,area,industry,list_date')
        if df.empty:
            all_stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,area,industry,list_date')
            df = all_stocks[all_stocks['name'].str.contains(keyword, case=False)]
    return df

# Fetch daily stock data and compute indicators
@st.cache_data
def get_stock_data(ts_code, start_date, end_date):
    """Fetch historical stock data and compute indicators"""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    df = pro.daily(ts_code=ts_code, start_date=start_str, end_date=end_str)  # Retrieve daily price data
    if df.empty:
        return None
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')  # Convert date format
    df = df.sort_values('trade_date')  # Sort by date
    df.set_index('trade_date', inplace=True)  # Set date as index
    calculate_technical_indicators(df)  # Compute indicators
    return df

# Compute technical indicators using TA-Lib
def calculate_technical_indicators(df):
    """Calculate MACD, RSI, KDJ, MA, BOLL using TA-Lib"""
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
    df['j'] = 3 * df['k'] - 2 * df['d']  # J line of KDJ
    # Moving Averages
    df['ma5'] = talib.SMA(df['close'].values, timeperiod=5)
    df['ma10'] = talib.SMA(df['close'].values, timeperiod=10)
    df['ma20'] = talib.SMA(df['close'].values, timeperiod=20)
    df['ma60'] = talib.SMA(df['close'].values, timeperiod=60)
    # Bollinger Bands
    df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(
        df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    return df


# Train a simple logistic regression model using technical indicators
def train_ai_model(df):
    """Train a logistic regression model to predict if next day's close > today"""
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # Create binary target

    # Select features derived from technical indicators
    features = ['macd', 'macdsignal', 'macdhist', 'rsi', 'k', 'd', 'j',
                'ma5', 'ma10', 'ma20', 'upperband', 'middleband', 'lowerband']

    # Drop rows with NaN in features or target
    df = df.dropna(subset=features + ['target'])

    if len(df) < 30:  # Ensure enough data for model training
        return None, "样本量不足，无法训练模型"

    X = df[features]  # Feature matrix
    y = df['target']  # Target labels

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate accuracy on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, f"模型准确率: {accuracy:.2f}"

# Generate AI-based trading suggestion using the trained model
def generate_ai_recommendation(model, latest_data):
    """Generate buy/sell recommendation based on latest technical features"""
    if model is None:
        return "无法生成建议: 模型未训练"

    features = ['macd', 'macdsignal', 'macdhist', 'rsi', 'k', 'd', 'j',
                'ma5', 'ma10', 'ma20', 'upperband', 'middleband', 'lowerband']

    if latest_data is None or len(latest_data) < 1:
        return "无法生成建议: 缺少数据"

    try:
        latest_features = latest_data[features].iloc[-1:]  # Get latest row of features
    except KeyError as e:
        return f"无法生成建议: 缺少特征 {e}"

    prediction = model.predict(latest_features)  # Predict class
    probability = model.predict_proba(latest_features)[0][prediction[0]]  # Get prediction confidence

    if prediction[0] == 1:
        return f"📈 AI建议: 买入 (概率: {probability:.2f})\n基于技术指标分析，预计股价将上涨"
    else:
        return f"📉 AI建议: 卖出 (概率: {probability:.2f})\n基于技术指标分析，预计股价将下跌"

# Plot candlestick chart with technical indicators
def plot_chart(df, stock_name):
    """Plot K-line chart and selected technical indicators using matplotlib"""
    if df is None or df.empty:
        st.warning("没有找到数据，请检查股票代码和时间范围")
        return

    # Create subplots depending on selected options
    fig, axes = plt.subplots(
        nrows=4 if show_volume else 3,
        ncols=1,
        figsize=(16, 12),
        gridspec_kw={
            'height_ratios': [3, 1, 1, 1] if show_volume else [3, 1, 1],
            'hspace': 0.5
        }
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    ax_idx = 0

    # K-line (candlestick chart)
    ax = axes[ax_idx]
    ax_idx += 1
    up = df[df.close >= df.open]  # Bullish candles
    down = df[df.close < df.open]  # Bearish candles
    col1 = 'red'
    col2 = 'green'

    # Plot up candles
    ax.bar(up.index, up.close - up.open, 0.8, bottom=up.open, color=col1)
    ax.bar(up.index, up.high - up.close, 0.2, bottom=up.close, color=col1)
    ax.bar(up.index, up.low - up.open, 0.2, bottom=up.open, color=col1)
    # Plot down candles
    ax.bar(down.index, down.close - down.open, 0.8, bottom=down.open, color=col2)
    ax.bar(down.index, down.high - down.open, 0.2, bottom=down.open, color=col2)
    ax.bar(down.index, down.low - down.close, 0.2, bottom=down.close, color=col2)

    # Plot moving averages
    if show_ma:
        ax.plot(df.index, df['ma5'], label='5日均线', color='orange', linewidth=1.5)
        ax.plot(df.index, df['ma10'], label='10日均线', color='purple', linewidth=1.5)
        ax.plot(df.index, df['ma20'], label='20日均线', color='blue', linewidth=1.5)
        ax.plot(df.index, df['ma60'], label='60日均线', color='brown', linewidth=1.5)
        ax.legend()

    # Plot Bollinger Bands
    if show_boll:
        ax.plot(df.index, df['upperband'], label='上轨', color='gray', linestyle='--', linewidth=1)
        ax.plot(df.index, df['middleband'], label='中轨', color='gray', linewidth=1)
        ax.plot(df.index, df['lowerband'], label='下轨', color='gray', linestyle='--', linewidth=1)
        ax.fill_between(df.index, df['upperband'], df['lowerband'], color='gray', alpha=0.1)

    ax.set_title(f'{stock_name} K线图', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)

    # MACD subplot
    if show_macd and ax_idx < len(axes):
        ax = axes[ax_idx]
        ax_idx += 1
        ax.plot(df.index, df['macd'], label='MACD', color='blue', linewidth=1.5)
        ax.plot(df.index, df['macdsignal'], label='Signal', color='red', linewidth=1.5)
        ax.bar(df.index, df['macdhist'], label='Histogram', color='gray', alpha=0.5)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title('MACD指标', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)

    # RSI and KDJ subplot
    if (show_rsi or show_kdj) and ax_idx < len(axes):
        ax = axes[ax_idx]
        ax_idx += 1
        if show_rsi:
            ax.plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=1.5)
            ax.axhline(70, color='red', linestyle='--', linewidth=0.8)
            ax.axhline(30, color='green', linestyle='--', linewidth=0.8)
        if show_kdj:
            ax.plot(df.index, df['k'], label='K', color='blue', linewidth=1.5)
            ax.plot(df.index, df['d'], label='D', color='orange', linewidth=1.5)
            ax.plot(df.index, df['j'], label='J', color='purple', linewidth=1.5)
            ax.axhline(80, color='red', linestyle='--', linewidth=0.8)
            ax.axhline(20, color='green', linestyle='--', linewidth=0.8)
        ax.set_title('RSI/KDJ指标', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)

    # Volume subplot
    if show_volume and ax_idx < len(axes):
        ax = axes[ax_idx]
        ax_idx += 1
        up_volume = df[df.close >= df.open]['vol']
        down_volume = df[df.close < df.open]['vol']
        ax.bar(up_volume.index, up_volume, color=col1, label='上涨成交量')
        ax.bar(down_volume.index, down_volume, color=col2, label='下跌成交量')
        ax.set_title('成交量', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)  # Render plot in Streamlit

# Main Streamlit logic to run the app
def main():
    stock_df = search_stock(stock_code)  # Search for matching stocks

    if stock_df.empty:
        st.warning("未找到匹配的股票，请尝试其他关键词")
        return

    st.subheader("股票信息")
    st.dataframe(stock_df, use_container_width=True)  # Display basic stock info

    # Option to add to watchlist
    selected_stock = stock_df.iloc[0]
    if not any(s['ts_code'] == selected_stock['ts_code'] for s in st.session_state.watchlist):
        if st.button('📌 添加到自选股', key='add_watch'):
            st.session_state.watchlist.append(selected_stock.to_dict())
            st.success(f'已将 {selected_stock["name"]} 添加到自选股')
    else:
        st.info(f'{selected_stock["name"]} 已在自选股中')

    # Retrieve data for selected stock
    ts_code = selected_stock['ts_code']
    stock_name = selected_stock['name']
    with st.spinner(f"正在获取{stock_name}({ts_code})的行情数据..."):
        df = get_stock_data(ts_code, start_date, end_date)

    # Display summary metrics
    if df is not None and not df.empty:
        st.subheader(f"{stock_name}({ts_code}) 行情摘要")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("最新价格", f"{df['close'].iloc[-1]:.2f}")
        with col2:
            change = df['close'].iloc[-1] - df['close'].iloc[-2]
            pct_change = (change / df['close'].iloc[-2]) * 100
            st.metric("涨跌幅", f"{change:.2f} ({pct_change:.2f}%)", delta=f"{pct_change:.2f}%")
        with col3:
            st.metric("最高价", f"{df['high'].iloc[-1]:.2f}")
        with col4:
            st.metric("最低价", f"{df['low'].iloc[-1]:.2f}")
        with col5:
            st.metric("成交量", f"{df['vol'].iloc[-1]/10000:.2f}万手")

        # Plot charts
        st.subheader("技术分析图表")
        plot_chart(df, stock_name)

        # Show raw data
        with st.expander("查看原始数据"):
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Show AI suggestion if enabled
        if use_ai:
            st.subheader("AI投资建议")
            with st.spinner("正在训练AI模型..."):
                model, accuracy_msg = train_ai_model(df.copy())
            if model is not None:
                st.success(accuracy_msg)
                recommendation = generate_ai_recommendation(model, df)
                st.info(recommendation)
            else:
                st.warning(accuracy_msg)

# Run app
if __name__ == "__main__":
    main()
