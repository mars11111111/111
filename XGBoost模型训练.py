import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib 

# 加载数据集并打乱
df = pd.read_excel(r"data12.xlsx")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 定义转换函数
def convert_work_hours_group(hours):
    if hours=='[35~40]':
        return 0
    elif hours=='(40~48]':
        return 1
    elif hours=='(48~54]':
        return 2
    elif hours=='(54,105]':
        return 3
    else:
        return None

# 应用转换函数到工时分组列
df['工时分组'] = df['工时分组'].apply(lambda x: convert_work_hours_group(x) if pd.notna(x) else x)

# 正确地列出所有分类变量
features = [ 'A2',  'A3', 'A5', 'A6', 'B3','B4','B5','smokeG', 'exerciseG1', 'exerciseG2', 'exerciseG3',  '年龄',  '工龄','上岗时间', '工时分组','生活满意度', '抑郁症状级别', '睡眠状况','疲劳蓄积程度']

# 创建特征集 X
X = df[features]

# 从特征列表中移除目标变量和序号
features = [col for col in features if col not in ['职业紧张程度']]

# 重新定义数值特征和分类特征
numerical_features = ['年龄', '工龄', '上岗时间', 'B3']
categorical_features =  [col for col in X.columns if col not in numerical_features]

# 将数值特征转换为数值类型，并处理无法转换的数据
for col in numerical_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 对分类特征进行标签编码
label_encoder = LabelEncoder()
for col in categorical_features:
    df[col] = df[col].astype(str)  # 确保所有数据都是字符串类型
    df[col] = label_encoder.fit_transform(df[col])

# 对数值特征进行缩放
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 确保目标变量“职业紧张程度”为数值类型
df['职业紧张程度'] = pd.to_numeric(df['职业紧张程度'], errors='coerce')

# 删除目标变量中含有缺失值的行
df = df.dropna(subset=['职业紧张程度'])

# 定义训练和评估模型的函数
def train_and_evaluate(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    # 初始化 XGBoost 分类器
    model_xgb = XGBClassifier(
        verbosity=1,
        seed=42,
        n_jobs=-1,
        eval_metric='mlogloss',
    )

    # 设置调整后的参数网格
    param_grid = {
        'max_depth': [6],
        'learning_rate': [0.05],
        'n_estimators': [470],
        'colsample_bytree': [0.9],
        'subsample': [0.8]
    }

    # 使用网格搜索和交叉验证寻找最优参数
    grid_search = GridSearchCV(
        estimator=model_xgb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=10,
        n_jobs=-1,
        verbose=1
    )

    # 训练模型
    grid_search.fit(X_train, y_train)

    # 使用最优模型预测训练集和测试集
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # 计算准确率
    train_accuracy = np.mean(y_train == y_train_pred)
    test_accuracy = np.mean(y_test == y_test_pred)

    # 输出分类报告
    print("训练集分类报告：")
    print(classification_report(y_train, y_train_pred))
    print("测试集分类报告：")
    print(classification_report(y_test, y_test_pred))

    # 输出准确率信息
    print(f"训练集准确率：{train_accuracy * 100:.2f}%")
    print(f"测试集准确率：{test_accuracy * 100:.2f}%")
    print(f"准确率差异：{abs(train_accuracy - test_accuracy) * 100:.2f}%")
    print("-" * 50)

    # 输出最佳参数
    print("最佳参数：", grid_search.best_params_)

    return best_model

# 构建以职业紧张程度为目标变量的模型
y1 = df['职业紧张程度'].values
best_model_1 = train_and_evaluate(X, y1)

# 保存模型为pkl文件
joblib.dump(best_model_1, 'best_model.pkl')