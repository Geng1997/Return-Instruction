import os
import shlex
import subprocess
import sys

db = []
dbSet = set()


# 读文件名列表
def read_file_db(db_path):
    if os.path.isfile(db_path):
        with open(db_path, 'r+') as f:
            for line in f:
                _line = line.strip()
                db.append(_line)
                dbSet.add(_line)


# 保存文件名列表
def save_file_db(db_path):
    with open(db_path, 'w') as f:
        for filename in db:
            f.write(filename + '\n')


# 反汇编
def process(target_dir, save_to):
    for fpath, dirs, fs in os.walk(target_dir):
        total = len(fs)
        count = 0
        for f in fs:
            count = count + 1
            print('{0}/{1}'.format(count, total))
            if f.endswith('.exe') or f.endswith('.dll'):
                filename = os.path.join(fpath, f)
                if filename in dbSet:
                    print('Skip ' + filename)
                else:
                    print(filename)
                    db.append(filename)
                    dbSet.add(filename)
                    line_no = len(db)
                    cmd = '"D:\\IDA\\IDA 7.0\\ida.exe" -c -A -S"D:\\Work\\GetData\\dump_ret(1).idc ' \
                          + save_to + ' ' + str(line_no) + '" "' + filename + '"'
                    subprocess.run(shlex.split(cmd))


if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #     print("Usage: %s D:\\sample\\xp d:\\xp.bin d:\\files.txt" % sys.argv[0])
    # else:
    #     db_path = sys.argv[3]
    #     read_file_db(db_path)
    #     process(sys.argv[1], sys.argv[2])
    #     save_file_db(db_path)
    db_path = r'D:\Work\Bin_Path\Xp.txt'
    read_file_db(db_path)
    process(r'D:\AAAA\sample\train2\xp', r'D:\Work\Bin_Location\Xp.bin')
    save_file_db(db_path)
