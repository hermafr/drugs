import numpy.random as rnd

def k_folds(n, k):
    inds = rnd.permutation(n)
    folds =[]
    fold_size = n / k
    for i in range(k):
        begin = int(i * fold_size)
        end = int((i+1) * fold_size)
        validation = list(inds[begin:end])
        training = [inds[j] for j in range(n) if j < begin or j >= end]
        folds.append((training, validation))
    return folds
