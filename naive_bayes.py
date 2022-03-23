from __future__ import division
import pandas as pd
import numpy
from math import log
from helpful_scripts import (
    _getWordsCountInDictionary,
    _processChunk,
    _getClassProbability,
    _fillClassDataList,
    _predict,
    _addClassProbability,
    _writeToFile,
)
import time
from scipy.sparse import csr_matrix


def main():
    VOCAB_SIZE = _getWordsCountInDictionary()
    CLASS_NUM = 3
    class_data_ls = [[] for i in range(CLASS_NUM)]
    csr_rows = []
    class_ls = []
    chunksize = 3  # optimal chunk size 100

    s_time_dask = time.time()
    with pd.read_csv("trainingpoems.csv", chunksize=chunksize, header=None) as reader:
        for chunk in reader:
            last_column = chunk.iloc[:, -1]
            last_column_ls = last_column.values.tolist()
            _fillClassDataList(class_data_ls, last_column_ls)
            class_ls = class_ls + last_column_ls
            chunk.drop(chunk.columns[0], axis=1, inplace=True)  # dropping document id
            chunk.drop(
                chunk.columns[len(chunk.columns) - 1], axis=1, inplace=True
            )  # dropping last column
            chunk_csrs = _processChunk(chunk, class_data_ls, last_column_ls)
            csr_rows = csr_rows + chunk_csrs

    e_time_dask = time.time()
    print("Read time 1000: ", (e_time_dask - s_time_dask), "seconds")

    index = class_data_ls[0].getLs()[0]
    print(index.data)
    x = numpy.log((index.data * 5) / 4)
    print(x)
    print(numpy.sum(x))
    print(class_data_ls[0].probability)

    probability_ls = _getClassProbability(class_ls)

    _addClassProbability(class_data_ls, probability_ls)

    id_ls = []
    prediction_ls = []
    with pd.read_csv("trainingpoems.csv", chunksize=chunksize, header=None) as reader:
        for chunk in reader:
            (chunk_id_ls, chunk_predcition_ls) = _predict(
                class_data_ls, chunk, VOCAB_SIZE
            )
            id_ls = id_ls + chunk_id_ls
            prediction_ls = prediction_ls + chunk_predcition_ls

    _writeToFile(id_ls, prediction_ls)


main()
