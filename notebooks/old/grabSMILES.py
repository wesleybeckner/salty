from __future__ import print_function
import json
import os
import sys
import pandas as pd
import numpy as np

#grab our checkName code
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import salty

salts=pd.read_csv('../salty/data/density.csv',delimiter=',')
salts = salts.reset_index() #reset the index so our for loops work
salts['salt_SMILES'] = np.nan

###run salts through checkName
for a, b in enumerate(salts['salt_name']):        
    try:
        salts['salt_SMILES'][a] = salty.checkName(b)
    except: #note that single-atom ions return an error in pychem, if failed computations are indicated remove problem
        pass #entries in the cell above

pd.DataFrame.to_csv(salts, path_or_buf='../salty/data/salts_with_smiles.csv')
