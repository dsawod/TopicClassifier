import pandas as pd

from helpful_scripts import _getWordsCountInDictionary, _processChunk
import time


def main():
    # VOCAB_SIZE = _getWordsCountInDictionary()

    csr_rows = []
    chunksize = 3
    s_time_dask = time.time()
    with pd.read_csv("trainingpoems.csv", chunksize=chunksize, header=None) as reader:
        for chunk in reader:
            chunk_csrs = _processChunk(chunk)
            csr_rows = csr_rows + chunk_csrs

    e_time_dask = time.time()
    print("Read time 1000: ", (e_time_dask - s_time_dask), "seconds")

    print("length of CSR matrices list", len(csr_rows))
    array0 = csr_rows[0].toarray()
    print(array0)


main()
