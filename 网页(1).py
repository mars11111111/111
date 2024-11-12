import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import xgboost as xgb
from matplotlib.font_manager import FontProperties

# 设置中文字体
font_path = "SIMSUNEXTG.TTF"  
font_prop = FontProperties(fname=font_path)

# 确保matplotlib使用指定的字体
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载模型
try:
    model = joblib.load('best_model.pkl')
except Exception as e:
    st.write(f"Error loading model: {e}")

# 获取模型输入特征数量及顺序
model_input_features = [ 'A2', 'A3', 'A5', 'B3', 'B4', 'B5', 'smokeG', 'exerciseG3', '年龄', '工龄', '工时分组', '生活满意度', '抑郁症状级别', '睡眠状况', '疲劳蓄积程度']
expected_feature_count = len(model_input_features)

# 定义新的特征选项及名称
cp_options = {
    0: '无症状 (0)',
    1: '轻度职业紧张 (1)',
    2: '中度职业紧张 (2)',
    3: '重度职业紧张 (3)'
}

# Streamlit 界面设置
st.title("职业紧张预测")

# 年龄输入
age = st.number_input("年龄：", min_value=1, max_value=120, value=50)

# 工龄输入
service_years = st.number_input("工龄：", min_value=1, max_value=120, value=50)

# 近一个月平均每天加班时间输入，对应 B3
overtime_hours = st.number_input("近一个月平均每天加班时间：", min_value=1, max_value=120, value=50)

# A2（性别）选择
A2_options = {1: '男性', 2: '女性'}
A2 = st.selectbox(
    "性别：",
    options=list(A2_options.keys()),
    format_func=lambda x: A2_options[x]
)

# A3（学历）选择
A3_options = {1: '初中及以下', 2: '高中或中专', 3: '大专或高职', 4: '大学本科', 5: '研究生及以上'}
A3 = st.selectbox(
    "学历：",
    options=list(A3_options.keys()),
    format_func=lambda x: A3_options[x]
)

# A5（月收入）选择
A5_options = {1: '少于 3000 元', 2: '3000 - 4999 元', 3: '5000 - 6999 元', 4: '7000 - 8999 元', 5: '9000 - 10999 元', 6: '11000 元及以上'}
A5 = st.selectbox(
    "月收入：",
    options=list(A5_options.keys()),
    format_func=lambda x: A5_options[x]
)

# B4（是否轮班）选择
B4_options = {1: '否', 2: '是'}
B4 = st.selectbox(
    "是否轮班：",
    options=list(B4_options.keys()),
    format_func=lambda x: B4_options[x]
)

# B5（是否需要上夜班）选择
B5_options = {1: '否', 2: '是'}
B5 = st.selectbox(
    "是否需要上夜班：",
    options=list(B5_options.keys()),
    format_func=lambda x: B5_options[x]
)

# smoke（是否吸烟）选择
smoke_options = {1: '是的', 2: '以前吸，但现在不吸了', 3: '从不吸烟'}
smoke = st.selectbox(
    "是否吸烟：",
    options=list(smoke_options.keys()),
    format_func=lambda x: smoke_options[x]
)

# 工时分组选择
working_hours_group_options = {1: '35 到 40 小时', 2: '40 到 48 小时', 3: '48 到 54 小时', 4: '54 到 105 小时'}
working_hours_group = st.selectbox(
    "工时分组：",
    options=list(working_hours_group_options.keys()),
    format_func=lambda x: working_hours_group_options[x]
)

# exercise（是否有进行持续至少 30 分钟的中等强度锻炼）选择
exercise_options = {1: '无', 2: '偶尔，1 - 3 次/月', 3: '有，1~3 次/周', 4: '经常，4~6 次/周', 5: '每天'}
exercise = st.selectbox(
    "是否有进行持续至少 30 分钟的中等强度锻炼：",
    options=list(exercise_options.keys()),
    format_func=lambda x: exercise_options[x]
)

# 生活满意度滑块
life_satisfaction = st.slider("生活满意度（1 - 5）：", min_value=1, max_value=5, value=3)

# 睡眠状况滑块
sleep_status = st.slider("睡眠状况（1 - 5）：", min_value=1, max_value=5, value=3)

# 疲劳积蓄程度滑块
work_load = st.slider("疲劳积蓄程度（1 - 5）：", min_value=1, max_value=5, value=3)

# 抑郁症状级别滑块
depression_level = st.slider("抑郁症状级别（1 - 5）：", min_value=1, max_value=5, value=3)

def predict():
    try:
        # 获取用户输入，并进行数据类型检查和转换
        user_inputs = {
            '年龄': int(age),
            'A2': int(A2),
            'A3': int(A3),
            'A5': int(A5),
            'B3': int(overtime_hours),
            'B4': int(B4),
            'B5': int(B5),
            'smokeG': int(smoke),
            'exerciseG3': int(exercise),
            '工龄': int(service_years),
            '工时分组': int(working_hours_group),
            '生活满意度': int(life_satisfaction),
            '抑郁症状级别': int(depression_level),
            '睡眠状况': int(sleep_status),
            '疲劳蓄积程度': int(work_load)
        }

        # 按照固定顺序整理特征值
        feature_values = [user_inputs[feature] for feature in model_input_features]
        features_array = np.array([feature_values])

        predicted_class = model.predict(features_array)[0]
        predicted_proba = model.predict_proba(features_array)[0]

        # 将数字对应转换为文本及格式化概率输出
        category_mapping = {'无职业紧张症状': 0, '轻度职业紧张症状': 1, '中度职业紧张症状': 2, '重度职业紧张症状': 3}
        predicted_category = [k for k, v in category_mapping.items() if v == predicted_class][0]
        probability_labels = ['无职业紧张症状', '轻度职业紧张症状', '中度职业紧张症状', '重度职业紧张症状']
        formatted_probabilities = [f'{prob:.4f}' for prob in predicted_proba]
        probability_output = [f"{label}: '{probability}'" for label, probability in zip(probability_labels, formatted_probabilities)]

        # 显示预测结果
        st.write(f"**预测类别：** {predicted_category}")
        st.write(f"**预测概率：** {dict(zip(probability_labels, formatted_probabilities))}")

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100
        advice = ""
        if predicted_category == '无职业紧张症状':
            advice = f"根据我们的模型，该员工无职业紧张症状。模型预测该员工无职业紧张症状的概率为 {probability:.2f}%。请继续保持良好的工作和生活状态。"
        elif predicted_category == '轻度职业紧张症状':
            advice = f"根据我们的模型，该员工有轻度职业紧张症状。模型预测该员工职业紧张程度为轻度的概率为 {probability:.2f}%。建议您适当调整工作节奏，关注自身身心健康。"
        elif predicted_category == '中度职业紧张症状':
            advice = f"根据我们的模型，该员工有中度职业紧张症状。模型预测该员工职业紧张程度为中度的概率为 {probability:.2f}%。建议您寻求专业帮助，如心理咨询或与上级沟通调整工作。"
        elif predicted_category == '重度职业紧张症状':
            advice = f"根据我们的模型，该员工有重度职业紧张症状。模型预测该员工职业紧张程度为重度的概率为 {probability:.2f}%。强烈建议您立即采取行动，如休假、寻求医疗支持或与管理层协商改善工作环境。"
        else:
            advice = "预测结果出现未知情况。"
        st.write(advice)
    
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)

        # 计算SHAP值并创建解释对象
        shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=model_input_features))
        shap_explanation = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=pd.DataFrame([feature_values], columns=model_input_features))

        # 绘制瀑布图，设置较大的 max_display 值
        shap.plots.waterfall(shap_explanation[0], max_display=len(user_inputs))
        plt.title('SHAP 值瀑布图')
        st.pyplot(plt.gcf())
    except Exception as e:
        st.write(f"Error in prediction: {e}")

if st.button("预测"):
    predict()