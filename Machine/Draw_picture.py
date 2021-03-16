import matplotlib.pyplot as plt
# 将获取的字符串转为float
from matplotlib import rcParams


def getNum(path):
    file = open(path, 'r')
    a = file.read()
    nstr = ''
    num = []
    for i in a:
        if i != ' ':
            nstr = nstr + i
        else:
            num.append(float(nstr))
            nstr = ''
    # print(num)
    return num


names = range(10, 95, 5)
names = [str(x) for x in list(names)]
x = range(len(names))

num_dt = getNum('Modify_DecisionTree_precision.txt')
num_rf = getNum('Modify_RF_precision.txt')
num_xgb = getNum('Modify_XGBoost_precision.txt')
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)
plt.plot(x, num_dt, '-')
plt.plot(x, num_rf, '--')
plt.plot(x, num_xgb, ':')
# plt.plot(num_svm, '.')
plt.xticks(x, names, rotation=1)
plt.legend(['Decision Tree', 'Random Forest', 'XGBoost'], loc='best')
# plt.legend(['DecisionTree', 'RF_ACC', 'XGBoost_ACC', 'SVM'], loc='best')
plt.subplots_adjust(bottom=0.10)
plt.xlabel('Training Set Size (% from dataset)')  # X轴标签
plt.ylabel("Accuracy")  # Y轴标签
plt.savefig('Modification_Machine_ACC.pdf')