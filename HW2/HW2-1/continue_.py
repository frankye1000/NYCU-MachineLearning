import numpy as np
from math import log 


def gussian(pixel_range, mean, var):
    p = 1 / (np.sqrt(2 * np.pi * var)) * np.exp(-(pixel_range - mean) ** 2 / (2 * var))
    return p


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
        label_probability_dict[label] = label_number_dict[label] / 60000

    # 目標: 使用高斯機率，算各別pixel條件機率，例:P(第一格,暗淡=25|label=0)=0.05,P(第一格,暗淡=85|label=0)=0.05,...,P(第七八四格, 暗淡=255|lebel=9)=0.01
    # 1.創一個三維 數量 矩陣儲存結果[10*784*256]，先計算train data每一個pixel的mean&var
    condiction_probability3D = np.zeros((10,784,256))
    for label in range(10):
        temp = train_data[train_label==label]  # 挑出各別label資料算各pixel mean&var
        mean = np.mean(temp, axis=0)
        var  = np.var(temp, axis=0)
        var[var==0] = 100000                   # 將var設大，這樣就讓"該pixel" "該灰階色調" 條件機率變小，降低影響
        for i in range(784):
            for pixel_range in range(256):
                condiction_probability3D[label][i][pixel_range] = gussian(pixel_range, mean[i], var[i])   # 高斯機率算每個pixel機率

    # 進行test資料集，連續版本運算
    error = 0
    for j in range(len(test_data)): 
        prob_sum = 0
        prob_logs = []
        for label in range(10):
            prob_log = 0
            for i, pixel in enumerate(test_data[j]):  
                pixel_range = pixel
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


    print("Imagination of numbers in Bayesian classifier: ")
    for label in range(10):
        temp = []
        for pixels in condiction_probability3D[label]:
            max_pixel_range_index = np.argmax(pixels)  # 選出淡到黑機率最大的pixel_range index
            if max_pixel_range_index >= 128:            # 因為>=128是1(黑) 
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