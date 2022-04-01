from math import log2
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, vstack
from scipy.special import softmax
import numpy, csv
import pandas as pd
import time


class ClassData:
    def __init__(self, class_id):
        self.class_id = class_id
        self.csr_ls = []
        self.probability = 0

    def getClassID(self):
        return self.class_id

    def addRow(self, row):
        self.csr_ls.append(row)

    def getLs(self):
        return self.csr_ls


def _predict(class_data_ls, chunk, total_words):
    size = len(chunk)
    id_ls = chunk.iloc[:, :1].values.tolist()
    predicted_class_ls = []
    chunk.drop(
        chunk.columns[0], axis=1, inplace=True
    )  # dropping document id from the data frame

    for i in range(size):
        class_probability_ls = [20]  # 20 classes
        row = chunk.iloc[i]  # This is Series
        row_as_list = row.values.tolist()  # Convert series to a list

        row_matrix = numpy.array(
            row_as_list
        )  # convert list to a matrix which is sparse
        csr = csr_matrix(row_matrix)  # Compress sparse row matrix
        for j in range(3):
            probability = _sentenceGivenClassProbability(
                class_data_ls[j], csr, total_words
            )
            print("\n")
            print(probability)
            class_probability_ls.append(probability)

        pred_class = class_probability_ls.index(max(class_probability_ls)) + 1
        predicted_class_ls.append(pred_class)

    return id_ls, predicted_class_ls


def _sentenceGivenClassProbability(class_data, csr, total_words):
    words_in_class, words_csr = _getNumOfUniqueWordsInClass(class_data)

    document_values = _getDocumentValues(words_csr, csr)

    beta = 1 / total_words
    class_probability = class_data.probability
    document_values_probabilites = numpy.log2(
        (document_values + beta) / (words_in_class + beta * total_words)
    )
    return log2(class_probability) + numpy.sum(document_values_probabilites)


def _getDocumentValues(words_csr, csr):

    words_indexes = words_csr.indices
    words_values = words_csr.data
    csr_indexes = csr.indices
    size = len(words_indexes)
    ls = []
    for i in range(size):
        index = words_indexes[i]
        if index in csr_indexes:
            ls.append(words_values[i])
    return numpy.array(ls)


def _addClassToClassDataObjects(class_data, ls):
    size = len(ls)
    for i in range(size):
        index = ls[i] - 1
        if class_data[index] == []:
            new_class_data = ClassData(index + 1)
            class_data[index] = new_class_data


def _addCSRsToClassDataObjects(class_data_ls, class_ls):
    size = len(class_ls)
    matrix = sparse.load_npz("CSR.npz")
    for i in range(size):
        index = class_ls[i] - 1
        object = class_data_ls[index]
        object.addRow(matrix[i])


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


def _addClassProbability(class_data_ls, pb_ls):
    size = len(pb_ls)
    for i in range(size):
        class_data_ls[i].probability = pb_ls[i]


# This method returns size of vocabulary
def _getWordsCountInDictionary():
    lines = []
    with open("vocabulary.txt") as file:
        lines = file.readlines()

    return len(lines)


def _getNumOfUniqueWordsInClass(class_data):
    csrrows_ls = class_data.getLs()
    size = len(csrrows_ls)
    sum = csrrows_ls[0]
    for i in range(1, size):
        sum = sum + csrrows_ls[i]
    return len(sum.data), sum


def _writeToFile(ids, predictions):
    with open("submit.csv", "w", newline="\n") as file:
        fieldnames = ["id", "class"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        size = len(ids)
        for x in range(size):
            writer.writerow({"id": ids[x], "class": predictions[x]})


def _readSparseMatrixFromCSV():
    chunksize = 100
    s_time_dask = time.time()
    df = pd.read_csv("training.csv", header=None, nrows=chunksize)
    df.drop(df.columns[0], axis=1, inplace=True)
    last_column = df.iloc[:, -1]
    class_ls = last_column.values.tolist()
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)

    matrix1 = csr_matrix(df.values)

    skip = [x for x in range(chunksize)]

    with pd.read_csv(
        "training.csv", header=None, skiprows=skip, chunksize=chunksize
    ) as reader:
        for chunk in reader:
            chunk.drop(chunk.columns[0], axis=1, inplace=True)
            last_column = chunk.iloc[:, -1]
            last_column_ls = last_column.values.tolist()
            class_ls = class_ls + last_column_ls
            chunk.drop(chunk.columns[len(chunk.columns) - 1], axis=1, inplace=True)
            matrix2 = csr_matrix(chunk.values)
            matrix = vstack([matrix1, matrix2])
            matrix1 = matrix

    e_time_dask = time.time()
    print("Read time 100: ", (e_time_dask - s_time_dask), "seconds")

    # sparse.save_npz("poems.npz", matrix)

    _saveClassListToFile(class_ls)


def _saveClassListToFile(class_ls):
    with open("ClassList.txt", "w") as output:
        for item in class_ls:
            output.write(str(item))
            output.write("\n")


def _getClassList():
    ls = []
    with open("ClassList.txt") as file:
        for line in file:
            ls.append(line.rstrip())

    ls = list(map(int, ls))
    return ls


def _initiateDeltaMatrix(target):
    num_of_classes = 20
    classes = [i + 1 for i in range(num_of_classes)]
    delta_col = len(target)
    delta = numpy.zeros([num_of_classes, delta_col])
    for i in range(num_of_classes):
        class_num = classes[i]
        for j in range(delta_col):
            if class_num == target[j]:
                delta[i, j] = class_num

    return csr_matrix(delta)


def _optimizeWeights(W, X, delta, l_r, penalty):
    for i in range(500):
        predictions = _findYGivenXandW(X, W)
        W = W + l_r * ((delta - predictions) @ X - penalty * W)

    return W


def _findYGivenXandW(X, W):
    X_transposed = X.transpose()
    product = W @ X_transposed
    predictions = numpy.exp(product.toarray())

    return csr_matrix(predictions)


def _predict_LR(W):
    W_tranposed = W.transpose()
    pred_ls = []

    chunksize = 100
    with pd.read_csv("testing.csv", header=None, chunksize=chunksize) as reader:
        for chunk in reader:

            # chunk.drop(chunk.columns[0], axis=1, inplace=True)
            chunk_csr = csr_matrix(chunk)
            product = chunk_csr @ W_tranposed
            result = numpy.exp(product.toarray())
            predictions = numpy.argmax(result, axis=1)
            ls = predictions.tolist()
            pred_ls = pred_ls + ls
            print(pred_ls)

    return pred_ls
