# 时间：2024年5月16日  Date： May 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Yunfei Gui, Jinhuan Luo 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师

# coding=utf-8
# 训练模型并预测测试集中数据，计算测试集与真实之间的误差


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 加载数据集
train_dataSet = pd.read_csv('modified_数据集Time_Series448_detail.dat')
test_dataSet = pd.read_csv('modified_数据集Time_Series660_detail.dat')

# columns表示原始列，noise_columns表示添加噪声的额列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth', 'RECORD']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth', 'Error_RECORD']

# 划分训练集中X_Train和y_Train
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]

# 定义训练模型（修改部分）
model = RandomForestRegressor(n_jobs=-1) # n_jobs：调用CPU全部内核，运行效率更高
model.fit(X_train,y_train)

# 划分测试集中X_test和y_test
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 预测值
y_predict = model.predict(X_test)

results =[]
# 遍历y_test和y_predict，并且计算误差
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)

    # 格式化True_Value和Predicted_Value为原始数据格式
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))  # 修改ERROR数据格式
    results.append([formatted_true_value, formatted_predicted_value, formatted_error]) # 保存结果

# 结果写入CSV文件当中
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result.csv", index=False)
