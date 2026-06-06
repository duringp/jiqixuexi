from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression#线性回归
from sklearn.metrics import mean_squared_error
from sklearn import svm #引入svm包
from sklearn.ensemble import RandomForestClassifier#引入随机森林
from sklearn.tree import DecisionTreeClassifier, plot_tree#引入决策树
diabetes = datasets.load_diabetes()#加载糖尿病数据集
x = diabetes.data
y = diabetes.target

#将数据集拆分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#创建一个多元线性回归算法对象
lr = LinearRegression()
#创建svm分类器并进行训练：首先，利用sklearn中的SVC（）创建分类器对象，其中常用的参数有C（惩罚力度）、kernel（核函数）、gamma（核函数的参数设置）、decision_function_shape（因变量的形式），再利用fit()用训练数据拟合分类器模型。
'''C越大代表惩罚程度越大，越不能容忍有点集交错的问题，但有可能会过拟合（defaul C=1）；
kernel常规的有‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ ，默认的是rbf；
gamma是核函数为‘rbf’, ‘poly’ 和 ‘sigmoid’时的参数设置，其值越小，分类界面越连续，其值越大，分类界面越“散”，分类效果越好，但有可能会过拟合，默认的是特征个数的倒数；
decision_function_shape='ovr'时，为one v rest（一对多），即一个类别与其他类别进行划分，等于'ovo'时，为one v one（一对一），即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
'''
model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')

# 训练随机森林分类器
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0, oob_score=True)

# 训练决策树
clf = DecisionTreeClassifier(max_depth=3, random_state=0)

#使用训练集训练模型
clf.fit(x_train, y_train)
#使用测试集进行预测
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)


#打印模型的均方差
print("均方误差：%.2f" %mean_squared_error(y_train, y_pred_train))
print("均方误差：%.2f" %mean_squared_error(y_test, y_pred_test))