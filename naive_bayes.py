import pandas as pd

from helpful_scripts import _getWordsCountInDictionary, _processChunk, _getClassProbability, _getCountsPerClass
import time


def main():
    # VOCAB_SIZE = _getWordsCountInDictionary()

    csr_rows = []
    class_ls = []
    chunksize = 10 ** 2
    s_time_dask = time.time()
    # with pd.read_csv("trainingpoems.csv", chunksize=chunksize, header=None) as reader:
    #     for chunk in reader:
    #         last_column = chunk.iloc[:, -1]
    #         last_column_ls = last_column.values.tolist()
    #         class_ls = class_ls + last_column_ls
    #         chunk_csrs = _processChunk(chunk)
    #         csr_rows = csr_rows + chunk_csrs

    e_time_dask = time.time()
    # print("Read time 1000: ", (e_time_dask - s_time_dask), "seconds")
    #
    # print("length of CSR matrices list", len(csr_rows))
    # array0 = csr_rows[0].toarray()
    # print(array0)
    # print(csr_rows[0])
    # print(class_ls)
    #
    # probability_ls = _getClassProbability(class_ls)
    # print(probability_ls)

    df = pd.read_csv("training-poems.csv", header=None)
    _getCountsPerClass(df)

main()
