# This method returns a list of prior probabilities
def _getClassProbability(data):
    # If whole example passed
    # do list = data[-1]

    # we know we only have 20 classes
    list = [0] * 21
    size = len(data)
    for i in range(size):
        id = data[i]
        list[id] = list[id] + 1
    del list[0]
    probability_list = [x / size for x in list]

    return probability_list


# This method returns size of vocabulary
def _getWordsCountInDictionary():
    lines = []
    with open("vocabulary.txt") as file:
        lines = file.readlines()

    return len(lines)


# This method returns total words in a dictionary
def _getTotalWordsCount(document):
    # document is an instance in training file
    size = len(document)
    count = 0
    for i in range(size):
        count = count + document[i]

    return count
