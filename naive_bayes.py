import pandas as pd

from helpful_scripts import (
    _getWordsCountInDictionary,
    _processChunk,
    _getClassProbability,
)
import time
from scipy.sparse import csr_matrix


def main():
    # VOCAB_SIZE = _getWordsCountInDictionary()

    csr_rows = []
    class_ls = []
    chunksize = 3  # optimal chunk size 100
    ls = []
    s_time_dask = time.time()
    with pd.read_csv("trainingpoems.csv", chunksize=chunksize, header=None) as reader:
        for chunk in reader:
            last_column = chunk.iloc[:, -1]
            last_column_ls = last_column.values.tolist()
            class_ls = class_ls + last_column_ls
            chunk_csrs = _processChunk(chunk)
            csr_rows = csr_rows + chunk_csrs

    e_time_dask = time.time()
    print("Read time 50: ", (e_time_dask - s_time_dask), "seconds")


main()
