import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PredictionApp:
    def __init__(self, master):
        self.master = master
        self.model = None
        self.scaler = StandardScaler()
        self.n_in = 1
        self.n_out = 1
        self.n_vars = 0
        self.test_file = "test_data.xlsx"

        self.setup_ui()
        self.initialize_model()

    def series_to_supervised(self, data, n_in=1, n_out=1):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()

        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)
        return agg

    def attention_layer(self, inputs, time_steps):
        a = Permute((2, 1))(inputs)
        a = Dense(time_steps, activation='softmax')(a)
        a_probs = Permute((2, 1))(a)
        return Multiply()([inputs, a_probs])

    def initialize_model(self):
        try:
            self.log("正在初始化模型...")
            # 训练数据预处理
            dataset = pd.read_excel("expanded_data.xlsx")
            values = dataset.values.astype('float32')
            self.n_vars = values.shape[1]

            # 数据预处理
            reframed = self.series_to_supervised(values, self.n_in, self.n_out)
            contain_vars = [('var%d(t-%d)' % (j, i)) for i in range(1, self.n_in+1) for j in range(1, self.n_vars+1)]
            data = reframed[contain_vars + ['var8(t)']]
            values = data.values

            # 数据标准化
            self.scaler.fit(values)

            # 模型构建
            pop = loadmat('result/SSA_para - 副本.mat')['best'].reshape(-1,)
            alpha = pop[0]
            hidden_nodes0 = int(pop[1])

            inputs = Input(shape=(self.n_in, self.n_vars))
            lstm = LSTM(hidden_nodes0, activation='selu', return_sequences=True)(inputs)
            attention_out = self.attention_layer(lstm, time_steps=self.n_in)
            attention_flatten = Flatten()(attention_out)
            outputs = Dense(self.n_out)(attention_flatten)
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')

            # 模型训练
            train = self.scaler.transform(values)
            train_X, train_y = train[:, :self.n_in*self.n_vars], train[:, self.n_in*self.n_vars:]
            train_X = train_X.reshape((train_X.shape[0], self.n_in, self.n_vars))
            self.model.fit(train_X, train_y, epochs=10, batch_size=30, verbose=0)
            self.log("模型初始化完成！")
        except Exception as e:
            self.log(f"模型初始化失败：{str(e)}")
            messagebox.showerror("错误", f"模型初始化失败：{str(e)}")

    def setup_ui(self):
        self.master.title("直接电养护混凝土温度演变预测系统")
        self.master.geometry("1200x800")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('微软雅黑', 15))
        style.configure('TButton', font=('微软雅黑', 15), padding=10)
        style.configure('Result.TLabel', foreground='#2E86C1', font=('微软雅黑', 15 ,'bold'))

        main_paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # 左侧面板增强
        left_frame = ttk.Frame(main_paned, width=320)
        main_paned.add(left_frame)

        # 参数输入区域
        input_frame = ttk.LabelFrame(left_frame, text="实验参数设置", padding=20)
        input_frame.pack(pady=10, padx=10, fill=tk.X)

        params = [
            ("水灰比", "water-cement ratio", "0.3"),
            ("水泥占胶材比", "cement content", "1.0"),
            ("电压(V)", "voltage", "20"),
            ("砂率(%)", "sand rate", "40"),
            ("电极间的距离(mm)", "specimen length", "100"),
            ("初始温度(℃)", "initial temperature", "29.3")
        ]

        self.entries = {}
        for label_text, field, default in params:
            frame = ttk.Frame(input_frame)
            frame.pack(fill=tk.X, pady=3)
            ttk.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
            entry = ttk.Entry(frame)
            entry.insert(0, default)
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.entries[field] = entry

        # 新增结果展示区域
        result_frame = ttk.LabelFrame(left_frame, text="预测结果", padding=20)
        result_frame.pack(pady=10, padx=10, fill=tk.X)

        self.max_temp_var = tk.StringVar(value="峰值温度：-- ℃")
        ttk.Label(result_frame, textvariable=self.max_temp_var,
                  style='Result.TLabel').pack(pady=5)

        self.max_time_var = tk.StringVar(value="达峰时间：-- 分钟")
        ttk.Label(result_frame, textvariable=self.max_time_var,
                  style='Result.TLabel').pack(pady=5)

        # 操作按钮区域
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="生成预测文件", command=self.generate_test_data,
                   style='TButton').pack(pady=5, fill=tk.X)
        ttk.Button(btn_frame, text="执行温度预测", command=self.predict,
                   style='Accent.TButton').pack(pady=5, fill=tk.X)

        # 右侧显示区域
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame)

        # 图表区域
        chart_frame = ttk.LabelFrame(right_frame, text="温度预测结果", padding=20)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 日志区域
        log_frame = ttk.LabelFrame(right_frame, text="运行日志", padding=20)
        log_frame.pack(fill=tk.BOTH, padx=10, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, font=('Consolas', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def initialize_model(self):
        try:
            self.log("正在初始化模型...")
            # 训练数据预处理
            dataset = pd.read_excel("expanded_data.xlsx")
            values = dataset.values.astype('float32')
            self.n_vars = values.shape[1]

            # 数据预处理
            reframed = self.series_to_supervised(values, self.n_in, self.n_out)
            contain_vars = [('var%d(t-%d)' % (j, i)) for i in range(1, self.n_in + 1) for j in
                            range(1, self.n_vars + 1)]
            data = reframed[contain_vars + ['var8(t)']]
            values = data.values

            # 数据标准化
            self.scaler.fit(values)

            # 模型构建
            pop = loadmat('result/SSA_para - 副本.mat')['best'].reshape(-1, )
            alpha = pop[0]
            hidden_nodes0 = int(pop[1])

            inputs = Input(shape=(self.n_in, self.n_vars))
            lstm = LSTM(hidden_nodes0, activation='selu', return_sequences=True)(inputs)
            attention_out = self.attention_layer(lstm, time_steps=self.n_in)
            attention_flatten = Flatten()(attention_out)
            outputs = Dense(self.n_out)(attention_flatten)
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')

            # 模型训练
            train = self.scaler.transform(values)
            train_X, train_y = train[:, :self.n_in * self.n_vars], train[:, self.n_in * self.n_vars:]
            train_X = train_X.reshape((train_X.shape[0], self.n_in, self.n_vars))
            self.model.fit(train_X, train_y, epochs=10, batch_size=30, verbose=0)
            self.log("模型初始化完成！")
        except Exception as e:
            self.log(f"模型初始化失败：{str(e)}")
            messagebox.showerror("错误", f"模型初始化失败：{str(e)}")

    def generate_test_data(self):
        try:
            data = {field: float(entry.get()) for field, entry in self.entries.items()}
            
            # 创建测试数据
            time = np.arange(1, 601)
            df = pd.DataFrame({'time': time})
            for key, value in data.items():
                df[key] = value
            df['Temperature'] = 30.0
            
            df.to_excel(self.test_file, index=False)
            self.log("成功生成预测数据：test_data.xlsx")
            messagebox.showinfo("成功", "预测数据生成成功！")
        except Exception as e:
            self.log(f"生成预测数据失败：{str(e)}")
            messagebox.showerror("错误", f"参数输入错误：{str(e)}")

    def predict(self):
        try:
            self.log("开始温度预测...")
            # 读取测试数据
            dataset = pd.read_excel(self.test_file)
            values = dataset.values.astype('float32')
            
            # 数据预处理
            reframed = self.series_to_supervised(values, self.n_in, self.n_out)
            contain_vars = [('var%d(t-%d)' % (j, i)) for i in range(1, self.n_in+1) for j in range(1, self.n_vars+1)]
            data = reframed[contain_vars + ['var8(t)']]
            test_values = data.values
            
            # 数据标准化
            test_scaled = self.scaler.transform(test_values)
            test_X = test_scaled[:, :self.n_in*self.n_vars]
            test_X = test_X.reshape((test_X.shape[0], self.n_in, self.n_vars))

            # 执行预测
            yhat = self.model.predict(test_X)

            # 逆标准化
            yhat_inv = self.scaler.inverse_transform(
                np.hstack((test_X.reshape((test_X.shape[0], -1)), yhat)))[:, -self.n_out:]

            # 计算最高温度和时间
            max_temp = np.max(yhat_inv)
            max_time = np.argmax(yhat_inv) + 1  # 时间从1开始

            # 更新结果展示
            self.max_temp_var.set(f"峰值温度：{max_temp:.2f} ℃")
            self.max_time_var.set(f"达峰时间：{max_time} 分钟")

            # 更新图表
            self.ax.clear()
            self.ax.plot(yhat_inv[:, 0], color='#FF6F61', linestyle="-",
                         linewidth=1.5, label='预测温度')
            self.ax.set_title("直接电养护混凝土温度演变预测结果", fontsize=12)
            self.ax.set_xlabel("时间（min）", fontsize=10)
            self.ax.set_ylabel("温度（℃）", fontsize=10)
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            self.canvas.draw()
            self.log("温度预测完成！")
            
        except Exception as e:
            self.log(f"预测过程中发生错误：{str(e)}")
            messagebox.showerror("错误", f"预测失败：{str(e)}")

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.iconbitmap('favicon.ico')
    root.mainloop()