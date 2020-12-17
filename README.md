# Secondary structure prediction: a probabilistic and machine learning showdown

This repository is intended as supplementary material for the Laboratory of bioinformatics 2 @ UniBO.

## Scripts
- `preparejpred.py` and `preparetest.py` each execute the pipeline that generated the profiles from the respective datasets.
The profiles will be places under `data/training/profile` for jpred and `data/test/profile` for the test set.
They both also dump in their respective locations a summary for each dataset.
For the complete generation of the test set, `sampletest.py` offers the code to sample 150 sequences from the test set.
It may take a long time since it tries to match JPred original secondary structure proportions within a 2% tolerance limit.
However, the prepared and ready-to-go dataset are also already present, so there is no need to execute the pipeline if not for reproducibility.
These dumps can be found also in their respective folders.

- `gorpredictor.py` contains a single class, namely `GORmodel` that seamlessly interfaces with scikit-learn.
Implements `fit` `predict` `predict_single` and `load_model`. It is intended to be used just like any other scikit-learns's model.
  Refer to the documentation for the details.
  
- `gor-train.py` and `gor-predict.py` are CLI-interfaces that implement the `GORmodel`. 
  They require just a list of IDs in txt format (one per line) and a location for the profiles.
  Profiles can be generated through the auxillary function `generate_profile`, loadable from `profile_tools.py`.

- `performance-svm.py` and `performance-gor.py` generate the scores for the two different estimators.

- `svm_gpu.py` implements the grid search for the SVM. Beware of long computation times and the necessity to have a CUDA compliant GPU.

## Models
The results of the grid search can be found here.

## Report
The report, in latex format and pdf.

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