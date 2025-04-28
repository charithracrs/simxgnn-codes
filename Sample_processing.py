import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
import torch


def get_DTI(dti_path):
    mat_dti = np.loadtxt(dti_path)
    index = np.where(mat_dti == 1)
    dti = np.ones(shape=(len(index[0]), 3))
    dti[:,0] = index[0]
    dti[:,1] = index[1]
    dti = dti.astype(int)
    return dti

def cross_k_folds(data,k_folds,num1, num2):
    pos_data = []
    neg_data = []
    data_adj = np.zeros(shape=(num1, num2))
    data_adj[data[:,0],data[:,1]] = data[:,2]
    pos_index = np.where(data_adj == 1)

    neg_index = np.where(data_adj == 0)
    result = random.sample(range(0, len(neg_index[0])), len(pos_index[0]))
    neg_selected = np.array([neg_index[0][result], neg_index[1][result]])
    count = 0
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
    for  fold, (train, test) in enumerate(kfold.split(pos_index[0], pos_index[1])):
        train_index_X = pos_index[0][train]
        train_index_Y = pos_index[1][train]
        test_index_X = pos_index[0][test]
        test_index_Y = pos_index[1][test]
        test_index = np.array([test_index_X,test_index_Y])
        test_index = torch.tensor(test_index)
        pos_data.append(test_index)

        neg_data.append(torch.tensor(neg_selected[:, count:count+len(test)]))
        count+= len(test)
    return pos_data, neg_data, torch.tensor(np.array(pos_index)), torch.tensor(np.array(neg_selected))


def gen_train_data(fold_num, pos_fold, neg_fold, drug_num, protein_num,device):
    all_index = np.arange(drug_num * protein_num)
    test_mask = np.ones(drug_num * protein_num)

    """Get the positive samples other than fold_num and combine them into the positive samples of the training set"""
    pos_index = []
    for i,k in enumerate(pos_fold):
        """Remove positive sample data from mask"""
        index_id = k[0, :] * protein_num + k[1, :]
        test_mask[index_id] = 0

        if i == fold_num:
            continue
        if len(pos_index) == 0:
            pos_index = pos_fold[i]
        else:
            pos_index = np.concatenate((pos_index, pos_fold[i]), axis=-1)

    """
    Remove negative samples from the test set from the mask
    """
    neg_removed = neg_fold[fold_num]
    neg_removed_id = neg_removed[0,:] * protein_num + neg_removed[1,:]
    test_mask[neg_removed_id] = 0

    """
    Generate train negative samples
    """
    id_n = np.where(test_mask == True)
    id_chosen = np.random.choice(id_n[0], pos_index.shape[1])
    drug_id = np.floor_divide(id_chosen, protein_num)
    protein_id = np.mod(id_chosen, protein_num)
    neg_index = np.array([drug_id, protein_id])

    index = np.concatenate((pos_index, neg_index), axis=-1)
    pos_labels = np.ones(pos_index.shape[1])
    neg_labels = np.zeros(neg_index.shape[1])
    labels = np.concatenate((pos_labels, neg_labels), axis=-1)

    index = torch.LongTensor(index)
    labels = torch.Tensor(labels)
    shuff_temp = np.arange(0, len(labels))
    random.shuffle(shuff_temp)
    index = index[:, shuff_temp].to(device)
    labels = labels[shuff_temp].to(device)

    return index.to(device), labels.to(device)
