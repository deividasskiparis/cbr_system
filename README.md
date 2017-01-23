# House Price Prediction CBR # {#index}

This is a Case Based Reasoning System developed in python. Designed for predicting house prices, but built adaptable
 for other applications.

Coursework for Advanced Machine Learning Topics (AMLT)  
Master of Artificial Intelligence  
UPC, Barcelona, 2016  

## Authors ##
* Deividas Skiparis
* Jérôme Charrier
* Simon Savornin
* Daniel Siqueira

## Structure of the delivered ZIP file ##
* dataFromKaggle - Folder containing data used for testing. train301.csv contains binarized dataset with 301 dimensions; train79.csv is an original dataset from kaggle. Both files have had outliers removed.
* Documentation - Folder containing all the documentation and function references. Use Documentation/Documentation.html to access it.
* kNNweights - Folder containing .csv of feature weights, used for weighted Retrieval.
* root folder - contains report, user manual, all the functions and packages. Their names are self-explanatory

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The system requries Python 2.* to be installed on your local machine. The code was tested and verified
to run Windows and Ubuntu machines.

### Installing

In order to install the CBR system, open command window in the location
 where the package was extracted and run the following line

```
python setup.py install
```

All the dependencies will be installed automatically.

To verify the installation was correct, try:

```
python demo.py
```

If the system runs correctly similar outcome will be displayed:

```
>python demo.py
#####    Demo run for CBR system    #####

Randomly selected test case id: 625
Test case label -  160000.0
Reused case label -  137178.58
Error -  137178.58 ( -0.18 %)
Test case retained? -  True
Case-base size before/after -  1459 / 1460
```


## Running the tests

To run the tests, which were performed to assess the CBR engine,
run:
```
python testing.py
```

The testing procedure will perform 2 test runs and will display progress along the way

```


Stage 1 started:  2017-01-23 20:41:21.785000
Iteration:  1 of 600
Distance metric:  MANH+W
Retention Strategy:  Regular
k :  2
Fold :  1  of  10
[################### ] 97% (116 of 119)
```

The testing procedure will generate 1 csv output file with results for Euclidean and Manhattan distances
and 1 csv output file for Eixample distance