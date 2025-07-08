import tushare as ts
import requests
import tkinter as tk
from tkinter import messagebox, Listbox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
import tkinter.font as tkFont

# 设置Tushare token
TUSHARE_TOKEN = 'f8635a89f62c17130e715107bb0b4020528ebd0a9719d185d3bc0692'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
FAVORITES_FILE = 'favorites.json'

def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_favorites(favs):
    with open(FAVORITES_FILE, 'w') as f:
        json.dump(favs, f)

def format_ts_code(code):
    if len(code) == 6 and code.isdigit():
        if code.startswith('6'):
            return code + '.SH'
        elif code.startswith(('0', '3')):
            return code + '.SZ'
    return code
from datetime import datetime, time
#交易时间判断
from datetime import datetime, time

def is_trading_time():
    now = datetime.now().time()
    return (time(9,30) <= now <= time(11,30)) or (time(13,0) <= now <= time(15,0))

def get_realtime_price_tushare(stock_code):
    try:
        df = ts.get_realtime_quotes(stock_code)
        if df is not None and not df.empty:
            row = df.iloc[0]
            name = row['name']
            price_str = row['price']
            pre_close_str = row['pre_close']

            if not price_str or float(price_str) == 0.0:
                status = "⚠️ Market closed or no price available"
                # 这里判断交易时间
                if not is_trading_time():
                    status = "⏰ 非交易时段"
                return f"{name} {status}"

            price = float(price_str)
            pre_close = float(pre_close_str) if pre_close_str else 0.0
            change = price - pre_close
            percent = (change / pre_close) * 100 if pre_close != 0 else 0

            # 如果不是交易时段，依然可以显示价格，但加上提示
            if not is_trading_time():
                return f"{name} Current Price: {price:.2f} CNY Change: {change:+.2f} ({percent:+.2f}%) ⏰ 非交易时段"

            return f"{name} Current Price: {price:.2f} CNY Change: {change:+.2f} ({percent:+.2f}%)"

        return "⚠️ No real-time data returned"
    except Exception as e:
        return f"⚠️ Tushare real-time error: {e}"
 
def get_kline_data(ts_code, freq='D', limit=60):
    if not ts_code.endswith(('.SH', '.SZ')):
        ts_code = format_ts_code(ts_code)
    try:
        df = ts.pro_bar(ts_code=ts_code, adj='qfq', freq=freq, limit=limit)
        if df is None or df.empty:
            return None
        df = df.copy()
        df.sort_values('trade_date', inplace=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        return df
    except Exception as e:
        print(f"Error getting K-line data: {e}")
        return None

def signal_analysis(df):
    if len(df) < 2:
        return "Insufficient data"
    prev = df.iloc[-2]
    latest = df.iloc[-1]
    if pd.isna(prev['MA5']) or pd.isna(prev['MA20']) or pd.isna(latest['MA5']) or pd.isna(latest['MA20']):
        return "🔍 Insufficient MA data"
    if prev['MA5'] < prev['MA20'] and latest['MA5'] > latest['MA20']:
        return "📈 Golden Cross (Buy Signal)"
    elif prev['MA5'] > prev['MA20'] and latest['MA5'] < latest['MA20']:
        return "📉 Death Cross (Sell Signal)"
    else:
        return "🔍 No obvious signal"

def plot_chart(df, ts_code, frame, freq):
    for widget in frame.winfo_children():
        widget.destroy()

    # 1. 创建图表
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df['trade_date'], df['close'], label='Close Price', color='blue')
    ax.plot(df['trade_date'], df['MA5'], label='MA5', color='orange')
    ax.plot(df['trade_date'], df['MA20'], label='MA20', color='green')

    ax.set_title(f"{ts_code} Close Price Trend ({freq} Line)")

    # 2. 设置横坐标日期间隔（避免太挤）
    num_ticks = min(len(df), 8)
    step = max(1, len(df) // num_ticks)
    xticks = df['trade_date'][::step]
    ax.set_xticks(xticks)

    # 3. 根据周期格式化横坐标标签
    def format_date_label(date_obj):
     if freq == 'D':
         return date_obj.strftime('%Y-%m-%d')   # 2024-07-01
     elif freq == 'W':
        return date_obj.strftime('%b %Y')      # Jul 2024
     elif freq == 'M':
        return date_obj.strftime('%b %Y')      # Jul 2024
     else:
        return date_obj.strftime('%Y-%m-%d')


    ax.set_xticklabels([format_date_label(d) for d in xticks], rotation=45, ha='right')

    ax.legend()
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


class StockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("A-Share Stock Assistant (Styled)")
        self.root.configure(bg='#f5f7fa')

        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=11)

        top = tk.Frame(root, bg='#f5f7fa')
        top.pack(fill=tk.X, pady=10, padx=10)

        tk.Label(top, text="Stock Code:", bg='#f5f7fa').pack(side=tk.LEFT)
        self.entry = tk.Entry(top, width=12)
        self.entry.pack(side=tk.LEFT, padx=5)

        self.freq_var = tk.StringVar(value='D')
        tk.Label(top, text="Period:", bg='#f5f7fa').pack(side=tk.LEFT)
        tk.OptionMenu(top, self.freq_var, 'D', 'W', 'M', command=self.on_freq_change).pack(side=tk.LEFT, padx=5)

        tk.Button(top, text="Get", command=self.get_data, bg='#007acc', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="★ Favorite", command=self.add_to_favorites, bg='#ff9800', fg='white').pack(side=tk.LEFT)

        self.price_label = tk.Label(root, text="📈 Real-time Quote:", fg='darkgreen', bg='#f5f7fa', anchor='w')
        self.price_label.pack(fill=tk.X, padx=10)

        self.signal_label = tk.Label(root, text="📊 Signal:", fg='blue', bg='#f5f7fa', anchor='w')
        self.signal_label.pack(fill=tk.X, padx=10)
        
        self.date_label = tk.Label(root, text="📅 Latest Data Date:", fg='gray', bg='#f5f7fa', anchor='w')  # ←新增
        self.date_label.pack(fill=tk.X, padx=10)
    
        self.chart_frame = tk.Frame(root, bg='white', relief=tk.RIDGE, borderwidth=1)
        self.chart_frame.pack(fill=tk.BOTH, expand=1, padx=10, pady=5)

        fav_frame = tk.Frame(root, bg='#f5f7fa')
        fav_frame.pack(fill=tk.X, padx=10)
        tk.Label(fav_frame, text="My Favorites:", bg='#f5f7fa').pack(side=tk.LEFT)
        tk.Button(fav_frame, text="🗑 Delete", command=self.remove_favorite, bg='crimson', fg='white').pack(side=tk.RIGHT)

        self.favorites = load_favorites()
        self.listbox = Listbox(root, height=5)
        self.listbox.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.listbox.bind('<Double-Button-1>', self.select_favorite)
        self.refresh_favorites()

        self.auto_refresh_price()

    def on_freq_change(self, event=None):
        self.get_data()

    def get_data(self):
        code = self.entry.get().strip()
        if not code:
            messagebox.showwarning("Hint", "Enter a stock code")
            return
        ts_code = format_ts_code(code)
        stock_code_pure = ts_code[:6]
        self.price_label.config(text=get_realtime_price_tushare(stock_code_pure))
        df = get_kline_data(ts_code, self.freq_var.get())
        if df is None:
            messagebox.showerror("Error", "Failed to get data")
            self.signal_label.config(text="📊 Signal: Data error")
            return
        self.signal_label.config(text="📊 Signal: " + signal_analysis(df))
        plot_chart(df, ts_code, self.chart_frame, self.freq_var.get())
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.date_label.config(text=f"📅 Query Time: {now}")
    def add_to_favorites(self):
        code = self.entry.get().strip()
        ts_code = format_ts_code(code)
        if ts_code and ts_code not in self.favorites:
            self.favorites.append(ts_code)
            save_favorites(self.favorites)
            self.refresh_favorites()
            messagebox.showinfo("Added", f"{ts_code} added to favorites.")
        else:
            messagebox.showinfo("Hint", "Already in favorites.")

    def remove_favorite(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        code = self.listbox.get(idx)
        if messagebox.askyesno("Delete", f"Delete {code}?"):
            self.favorites.pop(idx)
            save_favorites(self.favorites)
            self.refresh_favorites()

    def refresh_favorites(self):
        self.listbox.delete(0, tk.END)
        for code in self.favorites:
            self.listbox.insert(tk.END, code)

    def select_favorite(self, event):
        sel = self.listbox.curselection()
        if sel:
            code = self.listbox.get(sel[0])
            self.entry.delete(0, tk.END)
            self.entry.insert(0, code)
            self.get_data()

    def auto_refresh_price(self):
        code = self.entry.get().strip()
        if code:
            stock_code_pure = format_ts_code(code)[:6]
            self.price_label.config(text=get_realtime_price_tushare(stock_code_pure))
        self.root.after(5000, self.auto_refresh_price)

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("850x600")
    app = StockApp(root)
    root.mainloop()
