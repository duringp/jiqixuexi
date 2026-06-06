# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 评估指标
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# 1. 加载数据集
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
df = pd.read_csv(url, names=col_names)

# 2. 缺失值处理
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in zero_features:
    med_1 = df[df['Outcome'] == 1][feature].median()
    med_0 = df[df['Outcome'] == 0][feature].median()
    df.loc[(df['Outcome'] == 1) & (df[feature] == 0), feature] = med_1
    df.loc[(df['Outcome'] == 0) & (df[feature] == 0), feature] = med_0

# 3. 划分数据 & 标准化
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. 训练多模型
models = {
    '逻辑回归 (LR)': LogisticRegression(),
    '决策树 (DT)': DecisionTreeClassifier(),
    '随机森林 (RF)': RandomForestClassifier(),
    '支持向量机 (SVM)': SVC(probability=True)
}

results = []
trained = {}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    trained[name] = model
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        '模型算法': name,
        '准确率': round(acc, 2),
        '精确率': round(prec, 2),
        '召回率': round(rec, 2),
        'F1分数': round(f1, 2),
        'AUC-ROC': round(auc, 2)
    })

res_df = pd.DataFrame(results)

#可视化 1 数据分布：患病 vs 健康
plt.figure(figsize=(6, 4))
counts = df['Outcome'].value_counts().sort_index()
plt.bar(counts.index, counts.values, color=['#1f77b4', '#ff7f0e'])
plt.xticks([0,1])
plt.xlabel('类别 (0=健康, 1=患病)')
plt.ylabel('人数')
plt.title('糖尿病类别分布')
plt.show()

#可视化 2 特征相关性热力图
plt.figure(figsize=(10,8))
corr = df.corr()
im = plt.imshow(corr, cmap='coolwarm')
plt.colorbar(im)
plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(np.arange(len(corr.columns)), corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", color="black")

plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()

#可视化 3 葡萄糖与糖尿病关系
plt.figure(figsize=(8,5))
g0 = df[df['Outcome']==0]['Glucose']
g1 = df[df['Outcome']==1]['Glucose']
plt.hist([g0, g1], bins=20, alpha=0.7, label=['健康','患病'])
plt.legend()
plt.xlabel('葡萄糖含量')
plt.ylabel('频数')
plt.title('葡萄糖含量与糖尿病关系')
plt.show()

#可视化 4 模型准确率对比柱状图
plt.figure(figsize=(10,5))
# 这里修复了！
plt.bar(res_df['模型算法'], res_df['准确率'], color='steelblue')
plt.ylim(0, 1)
plt.title('各模型准确率对比')
plt.ylabel('准确率')
plt.show()

#可视化 5 ROC 曲线
plt.figure(figsize=(8,6))
for name, model in trained.items():
    y_prob = model.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC 曲线')
plt.legend()
plt.show()

#可视化 6 特征重要性
rf = trained['随机森林 (RF)']
imp = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf.feature_importances_
}).sort_values('重要性', ascending=True)

plt.figure(figsize=(10,5))
plt.barh(imp['特征'], imp['重要性'], color='darkred')
plt.xlabel('重要性')
plt.title('特征重要性（随机森林）')
plt.show()
print("\n" + "="*50)
print("          糖尿病预测模型效果对比表")
print("="*50)
print(res_df.to_string(index=False))
print("="*50)