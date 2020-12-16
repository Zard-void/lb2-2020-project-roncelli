# Secondary structure prediction: a probabilistic and machine learning showdown

This repository is intended as supplementary material for the Laboratory of bioinformatics 2 @ UniBO.

To prepare the jpred profiles *de novo* run `preparejpred.py`.
If short on time, in `data/training` there is `jpred.joblib`, which is a dump of the training set obtained with window size set to 17.


To prepare the blind set, run `preparetest.py`, it will read from `pdb_raw_download.csv` and execute the pipeline from redundancy reduction to profile generation.
After that, it is possible to run `sampletest.py` to sample 150 random sequences. 
It may take a long time since it tries to match JPred original secondary structure proportions within a 2% tolerance limit.


`models` contains a single zip file with the fitted best estimator as a dump.

`scripts/gorprepredictor.py` is loadable estimator that seamlessly interfaces with scikit-learn. 
Implements `fit` `predict` `predict_single` and `load_model`.
The first three work exactly like scikit-learn estimators, refer to their documentation or read the doctstrings.
`load_model` loads a trained model in the form of a tsv file. This file can be generated through `gor-train.py`.
It's twin file `gor-predict.py` uses the same synthax as the training file; both are executable on the CLI. 
A file containing the sequence id to train/predict on, as well as the location of the relative profiles must be specified.


## Dependencies
```
biopython
biotite
pandas
scikit-learn
tqdm
numpy
joblib
thundersvm
sultan
mkdssp
```