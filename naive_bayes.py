import pandas as pd
import time
from helpful_scripts import _getWordsCountInDictionary
import dask.dataframe as dd
from scipy.sparse import csr_matrix


def main():
    # VOCAB_SIZE = _getWordsCountInDictionary()

    # time taken to read data
    s_time_dask = time.time()
    dask_df = dd.read_csv("training.csv")
    e_time_dask = time.time()

    print("Read with dask: ", (e_time_dask - s_time_dask), "seconds")

    print(dask_df.head(5))


main()
