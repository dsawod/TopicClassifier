import scipy.sparse
import numpy
from scipy.sparse import csr_matrix
from helpful_scripts import (
    _getClassList,
    _optimizeWeights,
    _initiateDeltaMatrix,
    _predict_LR,
    _writeToFile,
)


def main():
    X = scipy.sparse.load_npz("X.npz")
    # print(X)
    row, col = X.shape
    class_list = _getClassList()
    m = row  # num of examples
    k = 20  # num of classes
    n = col - 1  # number of attributes each example has
    learning_rate = 0.0001
    penalty = 0.9
    random = numpy.zeros((k, n + 1))
    W = csr_matrix(random)
    delta_matrix = _initiateDeltaMatrix(class_list)
    W = _optimizeWeights(W, X, delta_matrix, learning_rate, penalty)
    print("Weights optimized")
    print(W.data)
    id_ls, pred_ls = _predict_LR(W)
    _writeToFile(id_ls, pred_ls)


main()
