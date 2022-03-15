# This method processes the chucnk of training file read
def _processChunk(examples):
    classes = examples[-1]
    examples = examples.iloc[1:, :-1]
    size = len(examples)
    words_count_perclass = []
    for x in range(size):
        document = examples[x]
        total_words_in_document = _getTotalWordsCount(document)
        words_count_perclass.append(total_words_in_document)
    return words_count_perclass


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
