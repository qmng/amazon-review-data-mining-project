setupFiles.py creates train and test .csv files

crossSetup.py creates train and test .csv files for kFold with reviews_all.csv

to launch project.py with two models and a loss function:

python3 project.py <model1> <model2> <product_name> <lossFunc>

Possible parameters for models:

- base
- cluster
- average
- fusion

Possible parameters for product_name:

- tampax
- always
- oral-b
- pantene
- gillette
- all

Possible parameters for lossFunc:

- l2
- cross_entropy

kFold.py executes the k fold cross validation for the train and test sets of reviews_all.csv created by crossSetup.py

testCluster.sh script runs for all products individually with cluster model and writes results in *Result.txt file.

surfaceplot.py plots a 3d surface plot.
