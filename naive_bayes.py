from __future__ import division
import pandas as pd

from helpful_scripts import (
    _getWordsCountInDictionary,
    _getClassProbability,
    _addClassToClassDataObjects,
    _predict,
    _addClassProbability,
    _writeToFileNB,
    _getClassList,
    _addCSRsToClassDataObjects,
)
import time
from scipy.sparse import csr_matrix


def main():
    s_time_dask = time.time()

    VOCAB_SIZE = _getWordsCountInDictionary()
    # VOCAB_SIZE = 70
    CLASS_NUM = 20
    class_data_ls = [[] for i in range(CLASS_NUM)]
    class_ls = _getClassList()

    _addClassToClassDataObjects(class_data_ls, class_ls)

    _addCSRsToClassDataObjects(class_data_ls, class_ls)

    probability_ls = _getClassProbability(class_ls)

    _addClassProbability(class_data_ls, probability_ls)

    chunksize = 500
    id_ls = []
    prediction_ls = []
    with pd.read_csv("testing.csv", chunksize=chunksize, header=None) as reader:
        for chunk in reader:
            (chunk_id_ls, chunk_predcition_ls) = _predict(
                class_data_ls, chunk, VOCAB_SIZE
            )
            id_ls = id_ls + chunk_id_ls
            prediction_ls = prediction_ls + chunk_predcition_ls

    _writeToFileNB(id_ls, prediction_ls)
    e_time_dask = time.time()
    print("time taken to run : ", (e_time_dask - s_time_dask), "seconds")


main()
