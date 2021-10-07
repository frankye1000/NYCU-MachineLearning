import numpy as np
from math import log


def run(train_data, train_label, test_data, test_label):
    # 目標: 紀錄各label機率(P(0),P(1),....)
    # 1.創一個字典，紀錄各label數量
    label_number_dict = {}
    for label in train_label:
        if label in label_number_dict:
            label_number_dict[label] += 1
        else:
            label_number_dict[label] = 1


    # 2. 創一個字典，紀錄各label機率(P(0),P(1),....)
    label_probability_dict = {}
    for label in label_number_dict:
        label_probability_dict[label] = label_number_dict[label] / len(train_data)


    # 目標: 算個別條件機率，例:P(第一格,暗淡=10|label=0)=0.05,P(第一格,暗淡=8|label=0)=0.05,...,P(第七八四格,暗淡=30|lebel=9)=0.01
    # 1.創一個三維 數量 矩陣儲存結果[10*784*32]
    number3D = np.zeros((10, 784, 32))
    for i, label in enumerate(train_label):
        for k, pixel in enumerate(train_data[i]):
            # pixel_range代表黑色(0~32)淡深程度，
            pixel_range = pixel // 8
            number3D[label][k][pixel_range] += 1 
    
    # 2.創一個三維 條件機率 矩陣儲存結果[10*784*32]，例:label9 在第四格 是 淡黑(29)的機率
    condiction_probability3D = np.zeros((10, 784, 32), dtype=float)
    for label in range(10):
        for i in range(784):
            condiction_probability3D[label][i] = number3D[label][i] / label_number_dict[label]

    # 進行test資料集，離散版本運算
    error = 0
    for j in range(len(test_data)):
        prob_sum = 0
        prob_logs = []
        for label in range(10):
            prob_log = 0
            for i, pixel in enumerate(test_data[j]):  
                pixel_range = pixel // 8
                prob_log += log(max(1e-125, condiction_probability3D[label][i][pixel_range]))  # log x越接近0,y越接近負無限大,又因為log0不存在，所以將機率0設定一個非常接近零的數值當屏障 

            prob_logs.append(prob_log + log(label_probability_dict[label]))  # 最後要乘P(label)的機率
            prob_sum += prob_log + log(label_probability_dict[label])
        
        # 印出結果 #
        print("Postirior (in log scale):")
        log_result = np.array(prob_logs)/prob_sum
        for i, v in enumerate(log_result):
            print("{}: {}".format(i, v))
        pred = np.argmin(log_result)         # 因為開log，所以要選最小的才是機率最高的
        ans  = test_label[j] 
        print("Prediction: {}, Ans: {}".format(pred, ans))
        print()
        if pred != ans:
            error += 1
        
    # 計算錯誤率 #
    error_rate = error/len(test_data)
    print("Error rate: {}".format(error_rate))

    # 印出0/1數字 #
    print("Imagination of numbers in Bayesian classifier: ")
    for label in range(10):
        temp = []
        for pixels in condiction_probability3D[label]:
            max_pixel_range_index = np.argmax(pixels)  # 選出淡到黑機率最大的pixel_range index
            if max_pixel_range_index >= 16:            # 因為>=128("16"*8)是1(黑) 
                temp.append(1)
            else:
                temp.append(0)
        
        # 印出數字 #
        print("{}:".format(label))
        for i in range(len(temp)):
            if i != 0 and i % 28 == 0:
                print()
            print(temp[i], end="")
        print()