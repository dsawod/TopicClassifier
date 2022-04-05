# CS 429/529 Project 12
## Topic Categorization

## Details

Team Members:
- Mari Aoki
- Andres Lucero
- Devendra Sawod
- Todd Sipe

Details for the project are found here: 
[Kaggle NB](https://www.kaggle.com/c/cs529-project-2-nb/overview) /
[Kaggle LR](https://www.kaggle.com/competitions/cs529-project-2-lr/overview)


## Execution 

This code is written in Python and is platform independent.

To run this code on local machine:
- Make sure you have Python installed on your machine.
- Download the zip file from learn.
- Unzip zip file in the directory you want to run the code.
- Open up your terminal in the directory where the code is located.
  - To run Linear Regression:
    - First decide the learning rate and penalty you will want to use.
      - Go into the logistic_regression.py
      - On line 22 the variable learning_rate decides the learning rate.
        - set to desired value
      - On line 23 the variable penalty decides the penalty.
        - set to desired value
      - Save the changes you made in the logistic_regression.py file.
    - Run 'python logistic_regression.py'.
    - A csv file named 'submitLR.csv' will be created containing the prediction of each document.
    - In the terminal, the confusion matrix will also be outputted.
  - For Naive Bayes, there is no execution needed:
    - Look in report at part "Description of Naive Bayes Code" for details
