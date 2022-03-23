from math import log
from scipy.sparse import csr_matrix
import numpy, csv


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


# This method processes the chunk of training file read
# It returns list of compressed matrices of each row in the chunk.
def _processChunk(chunk, class_data_ls, class_ls):
    size = len(chunk)
    ls = []
    for i in range(size):
        row = chunk.iloc[i]  # This is Series
        row_as_list = row.values.tolist()  # Convert series to a list
        row_matrix = numpy.array(
            row_as_list
        )  # convert list to a matrix which is sparse
        compressed_row_matrix = csr_matrix(row_matrix)  # Compress sparse matrix
        index = class_ls[i] - 1
        class_data_ls[index].addRow(compressed_row_matrix)

        ls.append(compressed_row_matrix)

    return ls


def _predict(class_data_ls, chunk, total_words):
    size = len(chunk)
    id_ls = chunk.columns[0].values.tolist()
    predicted_class_ls = []
    chunk.drop(
        chunk.columns[0], axis=1, inplace=True
    )  # dropping document id from the data frame

    for i in range(size):
        class_probability_ls = [20]
        row = chunk.iloc[i]  # This is Series
        row_as_list = row.values.tolist()  # Convert series to a list
        row_matrix = numpy.array(
            row_as_list
        )  # convert list to a matrix which is sparse
        csr = csr_matrix(row_matrix)  # Compress sparse row matrix
        for j in range(20):
            class_probability_ls.append(
                _sentenceGivenClassProbability(class_data_ls[j], csr, total_words)
            )
        pred_class = class_probability_ls.index(max(class_probability_ls)) + 1
        predicted_class_ls.append(pred_class)

    return id_ls, predicted_class_ls


def _sentenceGivenClassProbability(class_data, csr, total_words):
    words_in_class = _getNumOfUniqueWordsInClass(class_data)
    document_values = csr.data
    prior = 1 / 12000
    class_probability = class_data.probability
    document_values_probabilites = numpy.log(
        (document_values + prior) / (words_in_class + prior * total_words)
    )
    return class_probability * numpy.sum(document_values_probabilites)


def _fillClassDataList(class_data, ls):
    size = len(ls)
    for i in range(size):
        index = ls[i] - 1
        if class_data[index] == []:
            new_class_data = ClassData(index + 1)
            class_data[index] = new_class_data


# This method returns a list of prior probabilities
def _getClassProbability(classes):
    # If whole example passed
    # do list = data[-1]

    # we know we only have 20 classes
    list = [0] * 3
    size = len(classes)
    for i in range(size):
        id = classes[i]
        list[id - 1] = list[id - 1] + 1

    probability_list = [x / size for x in list]

    return probability_list

def _addClassProbability(class_data_ls,pb_ls):
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
    return len(sum.data)


def _writeToFile(ids, predictions):
    with open("submit.csv", "w", newline="\n") as file:
        fieldnames = ["id", "class"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        size = len(ids)
        for x in range(size):
            writer.writerow({"id": ids[x], "class": prediction[x]})







