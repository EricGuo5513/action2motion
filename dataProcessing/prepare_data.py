import numpy as np

def batchLabelOneHot(batch_list, label_list):
    one_hot_matrix = np.zeros((len(batch_list), len(label_list)))
    for i in range(len(batch_list)):
        one_hot_matrix[i, label_list.index(batch_list[i])] = 1
    return one_hot_matrix

def singlelabelonehot(label, label_list):
    one_hot_matrix = np.zeros((1, len(label_list)))
    one_hot_matrix[0, label_list.index(label)] = 1
    return one_hot_matrix

# def singlelabelonehot(batch_list, label_list):
#     one_hot_matrix = np.zeros((len(batch_list), len(label_list)))
#     for i in range(len(batch_list)):
#         one_hot_matrix[i, label_list.index(batch_list[i])] = 1
#     return one_hot_matrix
