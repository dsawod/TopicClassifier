from scipy.sparse import csr_matrix
import numpy

# This method processes the chunk of training file read
# It returns list of compressed matrices of each row in the chunk.
def _processChunk(chunk_df):
    size = len(chunk_df)
    ls = []
    for i in range(size):
        row = chunk_df.iloc[i]  # This is Series
        row_as_list = row.values.tolist()  # Convert series to a list
        print(row_as_list)
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
    list = [0] * 3
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

"""
Takes in a Pandas dataframe with first element being the document ID, the last element being the class of the document, 
and the elements in between are the word counts of the document. 
Returns a list of lists. Each list contains the word counts for each class.
"""
def _getCountsPerClass(df):
    # Convert Pandas dataframe to list
    dfList = df.values.tolist()
    # maxNum stores the largest class number
    # Note: this uses the class number of the last document, but we are not guaranteed that the last document
    # will have the highest class number.
    maxNum = dfList[-1][-1]
    # [[] for i in range(maxNum)] creates a list of empty lists with the number of classes.
    # Note: using [[]]*maxNum doesn't work because each list will be pointing to the same one and causes problems later.
    # totalCounts first holds all the document word counts for each class.
    # Will later hold total word count for each class.
    totalCounts = [[] for i in range(maxNum)]
    # Group documents by class in totalCounts list
    for i in range(len(dfList)):
        row = dfList[i]
        classNum = row[-1]
        totalCounts[classNum - 1].append(row[1:-2])
    # For each class in totalCounts, add word count for indices in each document to get total word count.
    # Sum shortcut found from:
    # https://stackoverflow.com/questions/64264392/adding-numbers-in-the-same-indices-in-list-of-lists
    for i in range(len(totalCounts)):
        currentClass = totalCounts[i]
        totalCounts[i] = list(map(sum, zip(*currentClass)))
