def checkName(user_query):
    """
    checkName uses a database lookup to return either SMILES or IUPAC 
    names of salts given either one of those are provided as inputs.
    Default behavior is to return the SMILES encoding of a salt given
    the salt name as input.
    
    Parameters
    ------------------
    user_query : str
        string that will be used to query the database.
        
    Returns
    ------------------
    output: str
        either the name of the salt, cation, or anion; or SMILES of the
        salt, cation, or anion (SMILES for the salt are written as the 
        cation and ion SMILES strings separated by a comma)
    """
    ###function imports
    import os
    import pandas as pd
    import numpy as np
    
    ###Check to see that the database is present
    if os.path.isfile('../salty/data/saltInfo.csv') == False:
        print('database file missing... exiting')
        quit()
    df = pd.read_csv('../salty/data/saltInfo.csv').astype(str)

    try:
        target_lookup = df.loc[(df == user_query).any(axis=1),:]
        input_type = df.loc[:,(df == user_query).any(axis=0)].columns.values
        target_column_index = df.columns.get_loc(input_type[0])
    except:
        print("query %s not found" % target_lookup)
        return 0

    #row_index pairs 1-4, 2-5, and 3-6
    if target_column_index == 1 or target_column_index == 2 or target_column_index == 3:
        print("user has queried with a SMILES structure")
        target = target_lookup.iloc[0][target_column_index+3]
    else:
        print("user has queried with a name")
        target = target_lookup.iloc[0][target_column_index-3]
    print("your query has returned %s" % target)
    return target
