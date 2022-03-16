from scipy.sparse import csr_matrix
import numpy

# This method processes the chucnk of training file read
# It returns list of compressed matrices of each row in the chunk.
def _processChunk(chunk_df):
    size = len(chunk_df)
    ls = []
    for i in range(size):
        row = chunk_df.iloc[i]  # This is Series
        row_as_list = row.values.tolist()  # Convert series to a list
        row_matrix = numpy.array(
            row_as_list
        )  # convert list to a matrix which is sparse
        compressed_row_matrix = csr_matrix(row_matrix)  # Compress sparse matrix
        ls.append(compressed_row_matrix)

    return ls


# This method returns a list of prior probabilities
def _getClassProbability(classes):
    # If whole example passed
    # do list = data[-1]

    # we know we only have 20 classes
    list = [0] * 20
    size = len(classes)
    for i in range(size):
        id = classes[i]
        list[id - 1] = list[id - 1] + 1

    probability_list = [x / size for x in list]

    return probability_list


# This method returns size of vocabulary
def _getWordsCountInDictionary():
    lines = []
    with open("vocabulary.txt") as file:
        lines = file.readlines()

    return len(lines)


# This method returns total words in a document
def _getTotalWordsCount(document):
    # document is an instance in training file
    size = len(document)
    count = 0
    for i in range(size):
        count = count + document[i]

    return count
