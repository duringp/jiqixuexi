from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 加载数据（二分类）
X, y = datasets.make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 SVM
clf = svm.SVC(probability=True, random_state=42)  # probability=True 才能计算 AUC
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]  # AUC 需要概率

# 计算指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)  # 也叫召回率
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"准确率: {acc:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)  # 注意 y_train 此时是连续值
y_pred_reg = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_reg)
mae = mean_absolute_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)  # y_train 是 0/1
y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]
# 计算指标
acc = accuracy_score(y_test, y_pred_lr)
prec = precision_score(y_test, y_pred_lr)
rec = recall_score(y_test, y_pred_lr)  # 也叫召回率
f1 = f1_score(y_test, y_pred_lr)
auc = roc_auc_score(y_test, y_pred_proba_lr)

print(f"准确率: {acc:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")