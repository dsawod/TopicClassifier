import pandas as pd

from helpful_scripts import _getWordsCountInDictionary, _processChunk
import time


def main():
    # VOCAB_SIZE = _getWordsCountInDictionary()

    csr_rows = []
    chunksize = 10 ** 3
    s_time_dask = time.time()
    with pd.read_csv("training.csv", chunksize=chunksize) as reader:
        for chunk in reader:
            chunk_csrs = _processChunk(chunk)
            csr_rows.append(chunk_csrs)

    e_time_dask = time.time()
    print("Read time 1000: ", (e_time_dask - s_time_dask), "seconds")

    print(len(csr_rows))


main()
