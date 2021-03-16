import numpy as np

UNIT_LEN = 16 * 2 + 1 + 4 + 4  # 前后16字节+1标签字节 + 4字节location + 4字节文件ID


def read_data(filename):
    file = open(filename, "rb")
    data = np.fromfile(file, dtype=np.uint8)
    file.close()
    _len = data.size // UNIT_LEN
    # print(_len)
    data = data.reshape(_len, UNIT_LEN)
    # _data = data[:, 0:(UNIT_LEN - 1 - 4 - 4)] / 255
    _data = data[:, 0:(UNIT_LEN - 1 - 4 - 4)]
    # label = data[:, (UNIT_LEN - 1 - 4 - 4)] > 0
    label = data[:, (UNIT_LEN - 1 - 4 - 4)]
    a = data[:, (UNIT_LEN - 4)]
    b = data[:, (UNIT_LEN - 3)]
    c = data[:, (UNIT_LEN - 2)]
    d = data[:, (UNIT_LEN - 1)]
    file_id = np.zeros(_len, dtype=np.uint32)
    for i in range(_len):
        file_id[i] = a[i] + b[i] * 256 + c[i] * 256 * 256 + d[i] * 256 * 256 * 256

    a = data[:, (UNIT_LEN - 8)]
    b = data[:, (UNIT_LEN - 7)]
    c = data[:, (UNIT_LEN - 6)]
    d = data[:, (UNIT_LEN - 5)]
    _location = np.zeros(_len, dtype=np.uint32)
    for i in range(_len):
        _location[i] = a[i] + b[i] * 256 + c[i] * 256 * 256 + d[i] * 256 * 256 * 256

    i = 0
    for x in _location:
        if i < 2:
            print(format(x, '#04X'))
        else:
            break
        i = i + 1
    _data = np.c_[_data, _location]
    _data = np.c_[_data, file_id]
    return _data, label
