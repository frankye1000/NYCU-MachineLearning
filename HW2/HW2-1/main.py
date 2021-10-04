from loadMnist import load_images, load_labels
import numpy as np

train_data  = load_images("data/train-images.idx3-ubyte")
train_label = load_labels("data/train-labels.idx1-ubyte")
test_data   = load_images("data/t10k-images.idx3-ubyte")
test_label  = load_labels("data/t10k-labels.idx1-ubyte")

# 算個別label機率(P(0),P(1),....)
label_probability_dict = {}
for i in train_label:
    if i in label_probability_dict:
        label_probability_dict[i] += 1
    else:
        label_probability_dict[i] = 1

for i in label_probability_dict:
    label_probability_dict[i] = label_probability_dict[i] / 60000


# 算個別條件機率，例:P(第一格=0|label=0)=0.05,P(第一格=1|label=0)=0.05,...,P(第七八四格=lebel=1|9)=0.01
# 創一個三維機率矩陣儲存結果[10*784*2]，例:label9 在第四格 是 全黑(0)的機率
probability3D = re=np.zeros((10, 784, 2))
for i, label in enumerate(train_label):
    for k, data in enumerate(train_data[i]):
        if k == 0: # 代表黑色
            probability3D[label][k][0] += 1
        else:
            probability3D[label][k][1] += 1
    
print(probability3D[5])              
