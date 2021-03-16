# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# 将获取的字符串转为float
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


num_rnn = getNum('Modification_RNN_ACC.txt')
num_gru = getNum('Modification_GRU_ACC.txt')
num_lstm = getNum('Modification_LSTM_ACC.txt')
num_dense = getNum('Modification_Dense_ACC.txt')
# 准备数据
x = np.linspace(0, 100, 100)
y_1 = num_rnn
y_2 = num_gru
y_3 = num_lstm
y_4 = num_dense


# 绘图
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(x, y_1, linestyle=':')

ax.plot(x, y_2, linestyle='-')

ax.plot(x, y_3, linestyle='--')

ax.plot(x, y_4, linestyle='-.')

plt.legend(['TBRNN', 'TBGRU', 'TBLSTM', 'NN'], loc='best')

plt.xlabel('Epoch')  # X轴标签
plt.ylabel("Accuracy")  # Y轴标签

# 嵌入绘制局部放大图的坐标系
axins = inset_axes(ax, width="40%", height="30%", loc='center left',
                   bbox_to_anchor=(0.5, 0.1, 1, 1),
                   bbox_transform=ax.transAxes)

# 在子坐标系中绘制原始数据
axins.plot(x, y_1, ':')

axins.plot(x, y_2, '-')

axins.plot(x, y_3, '--')

# 设置放大区间
zone_left = 90
zone_right = 97

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.5  # x轴显示范围的扩展比例
y_ratio = 0.5  # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

# Y轴的显示范围
y = np.hstack((y_1[zone_left:zone_right], y_2[zone_left:zone_right], y_3[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

# 设置刻度的浮点格式，即保留小数点后几位
axins.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

# 建立父坐标系与子坐标系的连接线
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

# 显示
# plt.show()
plt.savefig('Modification_ALL_ACC.pdf', dpi=900)
