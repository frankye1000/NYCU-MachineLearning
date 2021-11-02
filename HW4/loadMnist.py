import struct
import numpy as np


def load_images(filepath):
    binfile = open(filepath, "rb")
    buffers = binfile.read()
    # 解析headers(16 bytes)，">"是big endian
    magic, num, rows, cols = struct.unpack_from(">iiii", buffers, 0)
    bits = num * rows * cols
    # 解析全部，從offset 0016 開始
    images = struct.unpack_from(">" + str(bits) + "B", buffers, offset=struct.calcsize(">iiii"))
    binfile.close()
    # 60000*784 or 10000*784 
    images = np.reshape(images, [num, rows * cols])
    return images


def load_labels(filepath):
    binfile = open(filepath, "rb")
    buffers = binfile.read()
    # 解析headers(8 bytes)
    magic, num = struct.unpack_from(">ii", buffers, 0)
    # 解析全部，從offset 0008 開始
    labels = struct.unpack_from(">" + str(num) + "B", buffers, offset=struct.calcsize(">ii"))
    binfile.close()

    labels = np.reshape(labels, [num])
    return labels


