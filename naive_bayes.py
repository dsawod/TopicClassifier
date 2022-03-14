import pandas as pd


def main():

    examples = []
    chunksize = 10 ** 3
    with pd.read_csv("training.csv", chunksize=chunksize) as reader:
        for chunk in reader:
            examples.append(chunk)

    print(len(examples))
    print(examples[0])


main()
