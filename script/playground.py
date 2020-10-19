import pandas as pd
import numpy as np
#marginal_residue = gor_model.loc['R' + str(window_position) + 'R', residue]
#marginal_secondary = gor_model.loc['R0' + secondary].sum()
window_size = 3
COLUMNS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
#df = pd.read_csv('/home/stefano/PycharmProjects/lb2-2020-project-roncelli/data/training/X_train.tsv',sep='\t', index_col=[0,1], header=[0,1], low_memory=False)

SS = ('A', 'B')

for s in SS:
    print(s)
#print(df.head())