import pandas as pd

from helpful_scripts import (
    _getWordsCountInDictionary,
    _processChunk,
    _getClassProbability,
    _getCountsPerClass,
    _fillClassDataList,
)
import time
from scipy.sparse import csr_matrix


def main():
    # VOCAB_SIZE = _getWordsCountInDictionary()
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

    # new = csr_rows[0] + csr_rows[1]
    # print(new)
    # print(new.indices)
    # print(new.data)

    data = class_data_ls[0].getLs()
    print(data)
    print(data[0])
    print(class_data_ls[0].getClassID())

    # probability_ls = _getClassProbability(class_ls)
    # print(probability_ls)


main()
