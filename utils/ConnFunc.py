import numpy as np 

def pearson(m1,m2):
    """
    Computes Correlation matrix 
    """
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
    return corr
