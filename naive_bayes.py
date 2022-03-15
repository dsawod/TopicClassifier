import pandas as pd
from helpful_scripts import _processChunk, _getWordsCountInDictionary


def main():
    VOCAB_SIZE = _getWordsCountInDictionary()
    classes = []
    document_words_count = []
    chunksize = 10 ** 3
    with pd.read_csv("training.csv", chunksize=chunksize) as reader:
        for chunk in reader:
            chunk_classes = chunk[-1]
            classes.append(chunk_classes)
            ls = _processChunk(chunk)
            document_words_count.append(ls)
     

main()
