# 股票行情分析平台

一个使用Python、Streamlit和Tushare API构建的股票行情分析工具，提供实时股票数据展示、技术指标分析和AI投资建议。

## ✨ 功能特点

- **股票搜索**：通过股票代码或名称快速搜索
- **多时间范围**：支持自定义选择股票数据的时间范围
- **技术指标**：展示K线图、MACD、RSI、KDJ、均线、布林带和成交量等多种技术指标
- **AI分析**：基于技术指标提供简单的买入/卖出建议
- **美观界面**：使用Streamlit构建的直观用户界面

## 🚀 安装步骤

### 1. 克隆项目

```bash
# 克隆仓库
git clone <repository-url>
cd stock_dashboard
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置Tushare API密钥

1. 访问[Tushare官网](https://tushare.pro/)注册账号并获取API密钥
2. 将项目根目录下的`.env.example`文件复制为`.env`
3. 编辑`.env`文件，填入您的Tushare API密钥：

```
TUSHARE_API_KEY=您的API密钥
```

## ▶️ 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中自动打开，默认地址为：http://localhost:8501

## 📊 技术指标说明

- **MACD**：移动平均收敛散度，用于判断股票价格趋势和动量
- **RSI**：相对强弱指数，衡量股票的超买超卖状态
- **KDJ**：随机指标，用于识别价格趋势的转折点
- **均线**：移动平均线，展示股票价格的趋势方向
- **布林带**：展示价格波动范围和 volatility
- **成交量**：展示股票交易的活跃程度

## ⚠️ 注意事项

- 本工具仅用于学习和研究目的，不构成任何投资建议
- 股票数据由Tushare API提供，需要有效的API密钥
- AI分析功能基于简单的技术指标模型，仅供参考
- 历史表现不代表未来收益，投资有风险

## 📁 项目结构

```
stock_dashboard/
├── app.py              # 主应用程序
├── requirements.txt    # 依赖库列表
├── .env.example        # 环境变量模板
└── README.md           # 项目说明文档
```
