import numpy as np
import tensorflow as tf
import shutil
from sklearn.model_selection import train_test_split
from GetData import read_data_NN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import time

start = time.clock()
tf.compat.v1.disable_eager_execution()

shutil.rmtree('./logs', ignore_errors=True)
X1, Y1 = read_data_NN.read_data(r'D:\ReturnInstruction\Bin_Location\x86.bin')
X2, Y2 = read_data_NN.read_data(r'D:\ReturnInstruction\Bin_Location\sample.bin')
X = np.vstack((X1, X2))
Y = np.vstack((Y1, Y2))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

location_test = x_test[:, 32:33]
print(np.shape(location_test))
file_id_test = x_test[:, 33:34]
X_train = x_train[:, 0:32]
print(np.shape(X_train))
X_test = x_test[:, 0:32]
print(np.shape(X_test))

Hex_test = X_test

# 转换标签形式
label_test = []
for label in y_test:
    # print(label)
    num = 0
    for x in label:
        if int(x) == 1:
            label_test.append(num)
            break
        else:
            num += 1

epoch = 100
batch_size = 10000

# 构建双层双向RNN模型
# model_rnn = tf.keras.Sequential()
# model_rnn.add(tf.keras.layers.Embedding(256, 32))
# model_rnn.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32, return_sequences=True)))
# model_rnn.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32)))
# model_rnn.add(tf.keras.layers.Dense(3, activation='softmax'))
#
# model_rnn.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#
# history_rnn = model_rnn.fit(X_train, y_train,
#                             epochs=epoch,
#                             batch_size=batch_size,
#                             validation_data=(X_test, y_test))
# model_rnn.save(r'D:\ReturnInstruction\NewResult\Modification_model_rnn2.h5')
#
# file = open(r'D:\ReturnInstruction\NewResult\Modification_RNN_ACC.txt', 'w')
# for l in history_rnn.history['val_accuracy']:
#     file.write(str(l) + ' ')
# file.close()
#
# # 构建预测集
# Y_pred_rnn = model_rnn.predict_classes(X_test)

# 构建双层双向LSTM模型
# model_lstm = tf.keras.Sequential()
# model_lstm.add(tf.keras.layers.Embedding(256, 32))
# model_lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)))
# model_lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
# model_lstm.add(tf.keras.layers.Dense(3, activation='softmax'))
#
# model_lstm.compile(loss='binary_crossentropy',
#                    optimizer='adam',
#                    metrics=['accuracy'])
#
# history_lstm = model_lstm.fit(X_train, y_train,
#                               epochs=epoch,
#                               batch_size=batch_size,
#                               validation_data=(X_test, y_test))
# model_lstm.save(r'D:\ReturnInstruction\NewResult\Modification_model_lstm.h5')
#
# # 获取每次迭代训练的预测结果
# file = open(r'D:\ReturnInstruction\NewResult\Modification_LSTM_ACC.txt', 'w')
# for l in history_lstm.history['val_accuracy']:
#     file.write(str(l) + ' ')
# file.close()
#
# # 构建预测集
# Y_pred_lstm = model_lstm.predict_classes(X_test)

# 构建双层双向GRU模型
model_gru = tf.keras.Sequential()
model_gru.add(tf.keras.layers.Embedding(256, 32))
model_gru.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True)))
model_gru.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)))
model_gru.add(tf.keras.layers.Dense(3, activation='softmax'))

model_gru.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history_gru = model_gru.fit(X_train, y_train,
                            epochs=epoch,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test))
model_gru.save(r'D:\ReturnInstruction\NewResult\Modification_model_gru.h5')

# 获取每次迭代训练的预测结果
file = open(r'D:\ReturnInstruction\NewResult\Modification_GRU_ACC.txt', 'w')
for l in history_gru.history['val_accuracy']:
    file.write(str(l) + ' ')
file.close()

# 构建预测集
Y_pred_gru = model_gru.predict_classes(X_test)
#
# Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# # 构建DENSE模型
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(256, activation='relu', input_dim=32))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(3, activation='softmax'))
# model.compile(loss='mse',
#               optimizer='nadam',
#               metrics=['accuracy'])
#
# history = model.fit(X_train, y_train,
#                     epochs=epoch,
#                     batch_size=batch_size,
#                     validation_data=(X_test, y_test))
# model.save(r'D:\ReturnInstruction\NewResult\Modification_model_Dense.h5')
#
# # 获取每次迭代训练的预测结果
# file = open(r'D:\ReturnInstruction\NewResult\Modification_Dense_ACC.txt', 'w')
# for l in history.history['val_accuracy']:
#     file.write(str(l) + ' ')
# file.close()
#
# # 构建预测集
# Y_pred_nn = model.predict_classes(X_test)


# 计算评价指标
# print("全连接神经网络评价指标")
# print(accuracy_score(Y_pred_nn, label_test))
# conf_mat = confusion_matrix(label_test, Y_pred_nn)
# print(conf_mat)
# print(classification_report(label_test, Y_pred_nn, digits=4))
# file = open(r'D:\ReturnInstruction\NewResult\Modification_Dense_report.txt', 'w')
# file.write(classification_report(label_test, Y_pred_nn, digits=4))
# file.close()
#
# print("RNN评价指标")
# print(accuracy_score(Y_pred_rnn, label_test))
# conf_mat = confusion_matrix(label_test, Y_pred_rnn)
# print(conf_mat)
# print(classification_report(label_test, Y_pred_rnn, digits=4))
# file = open(r'D:\ReturnInstruction\NewResult\Modification_RNN_report.txt', 'w')
# file.write(classification_report(label_test, Y_pred_rnn, digits=4))
# file.close()

# print("LSTM评价指标")
# print(accuracy_score(Y_pred_lstm, label_test))
# conf_mat = confusion_matrix(label_test, Y_pred_lstm)
# print(conf_mat)
# print(classification_report(label_test, Y_pred_lstm, digits=4))
# file = open(r'D:\ReturnInstruction\NewResult\Modification_LSTM_report.txt', 'w')
# file.write(classification_report(label_test, Y_pred_lstm, digits=4))
# file.close()

print("GRU评价指标")
print(accuracy_score(Y_pred_gru, label_test))
conf_mat = confusion_matrix(label_test, Y_pred_gru)
print(conf_mat)
print(classification_report(label_test, Y_pred_gru, digits=4))
file = open(r'D:\ReturnInstruction\NewResult\Modification_GRU_report.txt', 'w')
file.write(classification_report(label_test, Y_pred_gru, digits=4))
file.close()

endtime = time.clock() - start
print("总用时: " + str(endtime) + "s")