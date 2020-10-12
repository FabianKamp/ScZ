import numpy as np 

def pearson1(m1,m2): 
    m1 -= np.mean(m1, axis=-1, keepdims=True)
    norm_m1 = np.linalg.norm(m1, axis=-1)
    norm_m1[np.isclose(norm_m1,0)] = 1

    m2 -= np.mean(m2, axis=-1, keepdims=True)
    norm_m2 = np.linalg.norm(m2, axis=-1)
    norm_m2[np.isclose(norm_m2,0)] = 1 

    corr = np.matmul(m1,m2.T)/(norm_m1 * norm_m2)
    corr = np.diag(corr)
    return corr

def pearson2(m1, m2):
    corr=np.array([np.corrcoef((m1[i,:], m2[i,:]))[0,1] for i in range(len(m1))])
    return corr

def pearson3(m1,m2):
    n = m1.shape[1]

    m1 -= np.mean(m1, axis=-1, keepdims=True)
    std_m1 = np.std(m1, axis=-1, keepdims=True)
    std_m1[np.isclose(std_m1,0)] = 1
    m1 /= std_m1

    m2 -= np.mean(m2, axis=-1, keepdims=True)
    std_m2 = np.std(m2, axis=-1, keepdims=True)
    std_m2[np.isclose(std_m2,0)] = 1
    m2 /= std_m2

    corr = np.matmul(m1,m2.T)/n 
    corr = np.diag(corr)
    return corr
