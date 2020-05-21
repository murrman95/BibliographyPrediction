# Predict missing links in citation network

[INF554](https://moodle.polytechnique.fr/enrol/index.php?id=5622) Machine Learning I (2018-2019, M. Vazirgiannis, [Ecole Polytechnique](https://www.polytechnique.edu/)) - *Final project* - **Guillaume Dufau, Christopher Murray, Léon Zheng**

## Description

Edges have been deleted at random from a citation network. The objective is to reconstruct the initial network using graph and text information.

Please find in a pdf format the report for this project.

Here are the results of our team - **The stormtroopers** - top 23%:
[kaggle/evaluation](https://www.kaggle.com/c/inf554-2018/leaderboard)

## Getting Started

The project is composed of two Python files, main.py and main_kaggle.py. The former allows model testing and hyper-parameters tuning on the training set where the latter is constructing the predictions file for kaggle.

### Prerequisites

Libraries used:
* Scikit-learn
* NetworkX
* nltk
* matplotlib

## Authors

* **Guillaume Dufau** - [GuillaumeDufau](https://github.com/GuillaumeDufau)
* **Christopher Murray** - [murrman95](https://github.com/murrman95)
* **Leon Zheng** - [leonzheng2](https://github.com/leonzheng2)
