import pickle
import shutil
import numpy as np
from matplotlib import pyplot, rcParams
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics, preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# 针对机器学习重新改写的读取方式
from GetData import read_data_RF
import time

start = time.clock()
shutil.rmtree('./logs', ignore_errors=True)
# 数据集的位置
path = r"D:\Work\Bin_Location\x86.bin"
X1, Y1 = read_data_RF.read_data(r'D:\Work\Bin_Location\x86.bin')
X2, Y2 = read_data_RF.read_data(r'D:\Work\Bin_Location\sample.bin')
X = np.vstack((X1, X2))
Y = np.append(Y1, Y2)

# print("Decision Tree")
# DecisionTree_test = []
# for test_size in range(10, 95, 5):
#     test_size = 100 - test_size
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size/100, random_state=0)
#     print(test_size)
#     # Feature Scaling
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#
#     # Fitting Decision Tree Classification to the Training set
#     classifier = DecisionTreeClassifier(criterion='entropy', random_state=3)
#     classifier.fit(X_train, Y_train)
#
#     # Predicting the Test set results
#     predict_results = classifier.predict(X_test)
#     result = accuracy_score(predict_results, Y_test)
#     DecisionTree_test.append(result)
#     print(result)
#     if test_size == 10:
#         conf_mat = confusion_matrix(predict_results, Y_test)
#         print(conf_mat)
#         print(classification_report(Y_test, predict_results, digits=4))
#         file = open(r'D:\Work\NewResult\Modification_DecisionTree_report.txt', 'w')
#         file.write(classification_report(Y_test, predict_results, digits=4))
#         file.close()

print("Random Forest")
RF_test = []
for test_size in range(10, 95, 5):
    test_size = 100 - test_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size/100, random_state=0)
    print(test_size)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Decision Tree Classification to the Training set
    classifier = RandomForestClassifier(criterion='entropy', min_samples_leaf=2, class_weight='balanced', verbose=0)
    classifier.fit(X_train, Y_train)

    # Predicting the Test set results
    predict_results = classifier.predict(X_test)
    result = accuracy_score(predict_results, Y_test)
    RF_test.append(result)
    print(result)
    if test_size == 10:
        conf_mat = confusion_matrix(predict_results, Y_test)
        print(conf_mat)
        print(classification_report(Y_test, predict_results, digits=4))
        file = open(r'D:\Work\NewResult\Modification_RF_report.txt', 'w')
        file.write(classification_report(Y_test, predict_results, digits=4))
        file.close()
#
# print("XGBoost")
# XGBoost_test = []
# for test_size in range(10, 95, 5):
#     test_size = 100 - test_size
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size/100, random_state=0)
#     print(test_size)
#
#     # Feature Scaling
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#
#     # Feature Scaling
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#
#
#     model = XGBClassifier(learning_rate=0.01,
#                                objective='multi:softmax',
#                                n_estimators=100,          # 树的个数-10棵树建立xgboost
#                                max_depth=6,               # 树的深度
#                                min_child_weight=1,        # 叶子节点最小权重
#                                gamma=0.,                  # 惩罚项中叶子结点个数前的参数
#                                subsample=1,               # 所有样本建立决策树
#                                colsample_btree=1,         # 所有特征建立决策树
#                                scale_pos_weight=1,        # 解决样本个数不平衡的问题
#                                random_state=27,           # 随机数
#                                slient=0)                  # silent = 0 输出中间过程
#     model.fit(X_train, Y_train)
#     # 预测
#     predict_results = model.predict(X_test)
#     result = accuracy_score(predict_results, Y_test)
#     XGBoost_test.append(result)
#     print(result)
#     if test_size == 10:
#         conf_mat = confusion_matrix(predict_results, Y_test)
#         print(conf_mat)
#         print(classification_report(Y_test, predict_results, digits=4))
#         file = open(r'D:\Work\NewResult\Modification_XGBoost_report.txt', 'w')
#         file.write(classification_report(Y_test, predict_results, digits=4))
#         file.close()


# 记录不同训练集占比下的准确率
# file = open(r'D:\Work\NewResult\Modify_DecisionTree_precision.txt', 'w')
# for i in DecisionTree_test:
#     file.write(str(i) + ' ')
# file.close()
file = open(r'D:\Work\NewResult\Modify_RF_precision.txt', 'w')
for i in RF_test:
    file.write(str(i) + ' ')
file.close()
# file = open(r'D:\Work\NewResult\Modify_XGBoost_precision.txt', 'w')
# for i in XGBoost_test:
#     file.write(str(i) + ' ')
# file.close()
#
# names = range(10, 95, 5)
# names = [str(x) for x in list(names)]
# x = range(len(names))
#
#
# plt.plot(x, DecisionTree_test, '-', label='Decision Tree')
# plt.plot(x, RF_test, '--', label='Random Forest')
# plt.plot(x, XGBoost_test, '.', label='XGBoost')
# plt.legend()  # 让图例生效
# plt.xticks(x, names, rotation=1)
# # plt.show()
# config = {
#     "font.family": 'Times New Roman',  # 设置字体类型
# }
# rcParams.update(config)
# # plt.margins(0)
# plt.subplots_adjust(bottom=0.10)
# plt.xlabel('Training Set Size (% from dataset)')  # X轴标签
# plt.ylabel("Classification Accuracy")  # Y轴标签
# # pyplot.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1])
# plt.savefig(r'D:\Work\NewResult\Classification.jpg', dpi=900)

endtime = time.clock() - start
print("总耗时: " + str(endtime) + " s")