from __future__ import print_function
import pandas as pd
import os
import sys
import matplotlib.pylab as plt
import numpy as np
import itertools as it
from scipy.stats import uniform as sp_rand
from scipy.stats import mode
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from numpy.random import randint
import numpy.linalg as LINA
from sklearn.preprocessing import StandardScaler
def compareAlphas(datadf, saltlist, target, alpha_array=np.arange(1,0,-1e-1), SSF=0.5, tol_boot=1e-6, MSSI=20, \
                  tol_lasso=1e-10, wrapper=False, display=False):
    """
    hyperparameters are lambda (alpha), shuffle splitfraction (SSF), convergence critiera/allowance for(tol)
    maximum shuffle split iterations (MSSI)
    """
   
    data=np.array(datadf)
     
    print('Job will perform %s tests for lambda' % len(alpha_array))
    lambda_test_MSE_averages=[]
    n = data.shape[0]
    d = data.shape[1]
    d -= 1
    n_train = int(n*SSF) #set fraction of data to be for training
    n_test  = n - n_train
    
    X_train = np.zeros((n_train,d)) #prepare train/test arrays
    X_test  = np.zeros((n_test,d))
    Y_train = np.zeros((n_train))
    Y_test = np.zeros((n_test))
    X_train[:] = data[:n_train,:-1] #fill arrays according to train/test split
    Y_train[:] = np.log(data[:n_train,-1].astype(float))
    X_test[:] = data[n_train:,:-1]
    Y_test[:] = np.log(data[n_train:,-1].astype(float))
    
    #initiate dataframe   
    df = pd.DataFrame(np.log(datadf[target][n_train:].astype(float)))
    df["Temperature (K)"]=datadf["Temperature_K"][n_train:]
    df["Pressure (kPa)"]=datadf["Pressure_kPa"][n_train:]
    df["Salt Name"]=saltlist[n_train:]
    desSelected=[]

    for j in range(len(alpha_array)):
        
        ###Train the LASSO model
        model = Lasso(alpha=alpha_array[j],tol=tol_lasso,max_iter=2000)
        model.fit(X_train,Y_train)
        desSelected.append(model.coef_)
        df["Prediction for alpha of %s" % alpha_array[j]] = model.predict(X_test)
        
        ###Calculate test set MSE
        Y_hat =model.predict(X_test)
        n = len(Y_test)
        test_MSE = np.sum((Y_test-Y_hat)**2)**1/n
        lambda_test_MSE_averages=[].append(test_MSE)
        
        if wrapper==False:
            if (j+1/len(alpha_array)*10)%10 == 0:
                print('Job %s' % str(j+1/len(alpha_array)*10), r'% complete' )
    print('Job complete')
    return df, alpha_array, desSelected, lambda_test_MSE_averages

def checkName(user_query, index=False):
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
    ###Check to see that the database is present
    if os.path.isfile('../salty/data/saltInfo.csv') == False:
        print('database file missing... exiting')
        quit()
    df = pd.read_csv('../salty/data/saltInfo.csv').astype(str)

    try:
        target_lookup = df.loc[(df == user_query).any(axis=1),:]
        input_type = df.loc[:,(df == user_query).any(axis=0)].columns.values
        target_column_index = df.columns.get_loc(input_type[0])
        target_row_index = df.loc[(df == user_query).any(axis=1),:].index.tolist()[0]

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
    if index:
        return target, target_row_index
    else:
        return target
def wesCVLasso(datadf, alpha_array=np.arange(1,0,-1e-2), wrapper=False, tol_lasso=1e-10, display=False,\
              cv=5):
    """
    hyperparameters are lambda (alpha), bootstrap splitfraction (BSF), convergence critiera/allowance (tol)
    maximum number of bootstrap iterations (NBI)
    """
    print('Job will perform %s tests for lambda' % len(alpha_array))
    
    data=np.array(datadf)
    n = data.shape[0]
    d = data.shape[1]
    d -= 1
    n_train = int(n*(1-1/cv)) #split size according to cv value)
    n_test  = n - n_train
    for j in range(cv):
        data = np.random.permutation(data) #if you delete, will not be random ie separate by group
        X_train = np.zeros((n_train,d)) #prepare train/test arrays
        X_test  = np.zeros((n_test,d))
        Y_train = np.zeros((n_train))
        Y_test = np.zeros((n_test))
        X_train[:] = data[:n_train,:-1] #fill arrays according to train/test split
        Y_train[:] = np.log(data[:n_train,-1].astype(float))
        X_test[:] = data[n_train:,:-1]
        Y_test[:] = np.log(data[n_train:,-1].astype(float))
        averages=np.zeros(len(alpha_array))
        variances=np.zeros(len(alpha_array))

        for i in range(len(alpha_array)):
            ###Train the LASSO model
            model = Lasso(alpha=alpha_array[i],tol=tol_lasso,max_iter=2000)

            ###Calculate MSE
            scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
            avg = np.average(scores)
            std = np.std(scores)
            averages[i] = avg
            variances[i] = std
            optimum_alpha=alpha_array[np.argmin(np.absolute(averages))]
    if display:
        displayResult(alpha_array, averages, variances=variances, error=True)
        displayFeatures(optimum_alpha, tol_lasso, X_train, Y_train, data, datadf)
    print('Job complete')
    return averages, variances, alpha_array

def CVLasso(datadf, alpha_array=np.arange(1,0,-1e-2), wrapper=False, tol_lasso=1e-10, display=False):
    """
    hyperparameters are lambda (alpha), bootstrap splitfraction (BSF), convergence critiera/allowance (tol)
    maximum number of bootstrap iterations (NBI)
    """
    print('Job will perform %s tests for lambda' % len(alpha_array))
    
    data=np.array(datadf)
    n = data.shape[0]
    d = data.shape[1]
    d -= 1
    n_train = int(n*1) #note I've set this to 1, crossvalscore is taking care of the actual splits
    n_test  = n - n_train

    data = np.random.permutation(data) #if you delete, will not be random ie separate by group
    X_train = np.zeros((n_train,d)) #prepare train/test arrays
    X_test  = np.zeros((n_test,d))
    Y_train = np.zeros((n_train))
    Y_test = np.zeros((n_test))
    X_train[:] = data[:n_train,:-1] #fill arrays according to train/test split
    Y_train[:] = np.log(data[:n_train,-1].astype(float))
    X_test[:] = data[n_train:,:-1]
    Y_test[:] = np.log(data[n_train:,-1].astype(float))
    averages=np.zeros(len(alpha_array))
    variances=np.zeros(len(alpha_array))

    for i in range(len(alpha_array)):
        ###Train the LASSO model
        model = Lasso(alpha=alpha_array[i],tol=tol_lasso,max_iter=2000)
        
        ###Calculate MSE
        scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
        avg = np.average(scores)
        std = np.std(scores)
        averages[i] = avg
        variances[i] = std
        optimum_alpha=alpha_array[np.argmin(np.absolute(averages))]
        if wrapper==False:
            if (i+1/len(alpha_array)*10)%10 == 0:
                print('Job %s' % str(i+1/len(alpha_array)*10), r'% complete' )
    if display:
        displayResult(alpha_array, averages, variances=variances, error=True)
        displayFeatures(optimum_alpha, tol_lasso, X_train, Y_train, data, datadf)
    print('Job complete')
    return averages, variances, alpha_array

def CVLassoValidationSet(datadf, data, test_data, alpha_array=np.arange(1,0,-1e-2), wrapper=False,\
                         tol_lasso=1e-10, display=False):
    """
    CV WITH VALIDATION SET
    
    hyperparameters are lambda (alpha), bootstrap splitfraction (BSF), convergence critiera/allowance (tol)
    maximum number of bootstrap iterations (NBI)
    """
    print('Job will perform %s tests for lambda' % len(alpha_array))
    
    n = data.shape[0]
    d = data.shape[1]
    d -= 1
    n_train = int(n*1) #note I've set this to 1.
    n_test  = n - n_train

    data = np.random.permutation(data) #if you delete, will not be random ie separate by group
    X_train = np.zeros((n_train,d)) #prepare train/test arrays
    X_test  = np.zeros((n_test,d))
    Y_train = np.zeros((n_train))
    Y_test = np.zeros((n_test))
    X_train[:] = data[:n_train,:-1] #fill arrays according to train/test split
    Y_train[:] = np.log(data[:n_train,-1].astype(float))
    X_test[:] = data[n_train:,:-1]
    Y_test[:] = np.log(data[n_train:,-1].astype(float))
    averages=np.zeros(len(alpha_array))
    variances=np.zeros(len(alpha_array))

    for i in range(len(alpha_array)):
        ###Train the LASSO model
        model = Lasso(alpha=alpha_array[i],tol=tol_lasso,max_iter=2000)
        
        ###Calculate MSE
        scores = cross_val_score(model, X_train, Y_train, cv=20, scoring='neg_mean_squared_error')
        avg = np.average(scores)
        std = np.std(scores)
        averages[i] = avg
        variances[i] = std
        optimum_alpha=alpha_array[np.argmin(np.absolute(averages))]
        if wrapper==False:
            if (i+1/len(alpha_array)*10)%10 == 0:
                print('Job %s' % str(i+1/len(alpha_array)*10), r'% complete' )
    if display:
        displayResult(alpha_array, averages, variances=variances, error=True)
        displayFeatures(optimum_alpha, tol_lasso, X_train, Y_train, data, datadf)
    print('Job complete')
    return averages, variances, alpha_array


def bootstrapLasso(datadf, alpha_array=np.arange(1,0,-1e-2), BSF=0.5, tol_boot=1e-6, MBI=100, \
                  tol_lasso=1e-10, wrapper=False, display=False):
    """
    hyperparameters are lambda (alpha), bootstrap splitfraction (BSF), convergence critiera/allowance (tol)
    aximum bootstrap iterations (MBI)
    """
    data=np.array(datadf)
    print('Job will perform %s tests for lambda' % len(alpha_array))
    lambda_test_MSE_averages=[]
    for j in range(len(alpha_array)):
        test_MSE_array=[]
        for i in range(MBI):
            n = data.shape[0]
            d = data.shape[1]
            d -= 1
            n_train = int(n*BSF) #set fraction of data to be for training
            n_test  = n - n_train
            data = np.random.permutation(data) #if you delete, will not be random ie separate by group
            X_train = np.zeros((n_train,d)) #prepare train/test arrays
            X_test  = np.zeros((n_test,d))
            Y_train = np.zeros((n_train))
            Y_test = np.zeros((n_test))
            
            ###sample from training set with replacement
            for k in range(n_train):
                x = randint(0,n_train)
                X_train[k] = data[x,:-1]
                Y_train[k] = np.log(data[x,-1].astype(float))
            
            ###sample from test set with replacement
            for k in range(n_test):
                x = randint(n_train,n)
                X_test[k] = data[x,:-1]
                Y_test[k] = np.log(data[x,-1].astype(float))


            ###Train the LASSO model
            model = Lasso(alpha=alpha_array[j],tol=tol_lasso,max_iter=2000)
            model.fit(X_train,Y_train)

            
            ###Calculate test set MSE
            Y_hat =model.predict(X_test)
            n = len(Y_test)
            test_MSE = np.sum((Y_test-Y_hat)**2)**1/n
            test_MSE_array.append(test_MSE)
            if i > 0:
                conv_test = (np.average(test_MSE_array[:]) - np.average(test_MSE_array[:-1]))**2
                if conv_test < tol_boot:
                    break
            if i == MBI:
                print("%s lambda value did not converge" % alpha_array[j])
        lambda_test_MSE_averages.append(np.average(test_MSE_array))
        optimum_lambda = alpha_array[lambda_test_MSE_averages.index(min(lambda_test_MSE_averages))]
        if wrapper==False:
            if (j+1/len(alpha_array)*10)%10 == 0:
                print('Job %s' % str(j+1/len(alpha_array)*10), r'% complete' )
    if display:
        displayResult(alpha_array, lambda_test_MSE_averages)
        displayFeatures(optimum_lambda, tol_lasso, X_train, Y_train, data, datadf)
    print('Job complete')
    return lambda_test_MSE_averages, alpha_array


def bootstrapLassoInvisibleTest(datadf, data, test_data, alpha_array=np.arange(1,0,-1e-2), BSF=0.8, \
                                tol_boot=1e-10, MBI=100, tol_lasso=1e-10, wrapper=False, TSF=0.8,\
                               display=False):
    """
    BOOTSTRAP WITH COMPLETELY SEPARATE TEST SET
    
    hyperparameters are lambda (alpha), bootstrap splitfraction (BSF), convergence critiera/allowance (tol)
    maximum number of bootstrap iterations (MBI)
    """
    from sklearn.linear_model import Lasso
    print('Job will perform %s tests for lambda with %s reserved for validation and %s test set'\
          % (len(alpha_array), TSF, BSF))
    lambda_test_MSE_averages=[]
    for j in range(len(alpha_array)):
        test_MSE_array=[]
        for i in range(MBI):
            n = data.shape[0]
            d = data.shape[1]
            n2 = test_data.shape[0]
            d -= 1
            n_train = int(n*BSF) #set fraction of data to be for training
            n_test  = n2
            data = np.random.permutation(data) #if you delete, will not be random ie separate by group
            
            X_train = np.zeros((n_train,d)) #prepare train/test arrays
            X_test  = np.zeros((n_test,d))
            Y_train = np.zeros((n_train))
            Y_test = np.zeros((n_test))

            ###sample from training set with replacement
            for k in range(n_train):
                x = randint(0,n_train)
                X_train[k] = data[x,:-1]
                Y_train[k] = np.log(data[x,-1].astype(float))
            
            ###sample from test set with replacement
            for k in range(n_test):
                y = randint(0,test_data.shape[0])
                X_test[k] = test_data[y,:-1]
                Y_test[k] = np.log(test_data[y,-1].astype(float))

            ###Train the LASSO model
            model = Lasso(alpha=alpha_array[j],tol=tol_lasso,max_iter=2000)
            model.fit(X_train,Y_train)

            ###Calculate test set MSE
            Y_hat =model.predict(X_test)
            n = len(Y_test)
            test_MSE = np.sum((Y_test-Y_hat)**2)**1/n
            test_MSE_array.append(test_MSE)
            if i > 0:
                conv_test = (np.average(test_MSE_array[:]) - np.average(test_MSE_array[:-1]))**2
                if conv_test < tol_boot:
                    break
            if i == MBI:
                print("%s lambda value did not converge" % alpha_array[j])
        lambda_test_MSE_averages.append(np.average(test_MSE_array))
        if (j+1/len(alpha_array)*10)%10 == 0:# and wrapper==False:
            print('Job %s' % str(j+1/len(alpha_array)*10), r'% complete' )
    optimum_lambda = alpha_array[lambda_test_MSE_averages.index(min(lambda_test_MSE_averages))]
    print("Job complete, optimum lambda converged on %s" % optimum_lambda)
    if display:
        displayResult(alpha_array, lambda_test_MSE_averages)
        displayFeatures(optimum_lambda, tol_lasso, X_train, Y_train, data, datadf)
    return lambda_test_MSE_averages, alpha_array

def shuffleSplitLasso(datadf, alpha_array=np.arange(1,0,-1e-2), SSF=0.5, tol_boot=1e-6, MSSI=300, \
                  tol_lasso=1e-10, wrapper=False, display=False):
    """
    hyperparameters are lambda (alpha), shuffle splitfraction (SSF), convergence critiera/allowance (tol)
    maximum shuffle split iterations (MSSI)
    """
    data=np.array(datadf)
     
    print('Job will perform %s tests for lambda' % len(alpha_array))
    lambda_test_MSE_averages=[]
    for j in range(len(alpha_array)):
        test_MSE_array=[]
        for i in range(MSSI):
            n = data.shape[0]
            d = data.shape[1]
            d -= 1
            n_train = int(n*SSF) #set fraction of data to be for training
            n_test  = n - n_train
            data = np.random.permutation(data) #if you delete, will not be random ie separate by group
            X_train = np.zeros((n_train,d)) #prepare train/test arrays
            X_test  = np.zeros((n_test,d))
            Y_train = np.zeros((n_train))
            Y_test = np.zeros((n_test))
            X_train[:] = data[:n_train,:-1] #fill arrays according to train/test split
            Y_train[:] = np.log(data[:n_train,-1].astype(float))
            X_test[:] = data[n_train:,:-1]
            Y_test[:] = np.log(data[n_train:,-1].astype(float))

            ###Train the LASSO model
            model = Lasso(alpha=alpha_array[j],tol=tol_lasso,max_iter=2000)
            model.fit(X_train,Y_train)

            ###Calculate test set MSE
            Y_hat =model.predict(X_test)
            n = len(Y_test)
            test_MSE = np.sum((Y_test-Y_hat)**2)**1/n
            test_MSE_array.append(test_MSE)
            if i > 0:
                conv_test = (np.average(test_MSE_array[:]) - np.average(test_MSE_array[:-1]))**2
                if conv_test < tol_boot:
                    break
            if i == MSSI:
                print("%s lambda value did not converge" % alpha_array[j])
        lambda_test_MSE_averages.append(np.average(test_MSE_array))
        
        if wrapper==False:
            if (j+1/len(alpha_array)*10)%10 == 0:
                print('Job %s' % str(j+1/len(alpha_array)*10), r'% complete' )
    if display:
        displayResult(alpha_array, lambda_test_MSE_averages)
        displayFeatures(optimum_lambda, tol_lasso, X_train, Y_train, data, datadf)
    optimum_lambda = alpha_array[lambda_test_MSE_averages.index(min(lambda_test_MSE_averages))]
    print('Job complete')
    return lambda_test_MSE_averages, alpha_array

def shuffleSplitLassoInvisibleTest(datadf, data, test_data, alpha_array=np.arange(1,0,-1e-2), SSF=0.8, \
                                tol_boot=1e-10, MSSI=100, tol_lasso=1e-10, wrapper=False, display=False):
    """
    BOOTSTRAP WITH COMPLETELY SEPARATE TEST SET
    
    hyperparameters are lambda (alpha), shuffle split fraction (SSF), convergence critiera/allowance (tol)
    maximum number of shuffle split iterations (MSSI)
    """
    print('Job will perform %s tests for lambda' % len(alpha_array))
    lambda_test_MSE_averages=[]
    for j in range(len(alpha_array)):
        test_MSE_array=[]
        for i in range(MSSI):
            n = data.shape[0]
            d = data.shape[1]
            n2 = test_data.shape[0]
            d -= 1
            n_train = int(n*SSF) #set fraction of data to be for training
            n_test  = n2
            data = np.random.permutation(data) #if you delete, will not be random ie separate by group
            
            X_train = np.zeros((n_train,d)) #prepare train/test arrays
            X_test  = np.zeros((n_test,d))
            Y_train = np.zeros((n_train))
            Y_test = np.zeros((n_test))
            
            X_train[:] = data[:n_train,:-1] #fill arrays according to train/test split
            Y_train[:] = np.log(data[:n_train,-1].astype(float))
            X_test[:] = test_data[:,:-1]
            Y_test[:] = np.log(test_data[:,-1].astype(float))

            ###Train the LASSO model
            model = Lasso(alpha=alpha_array[j],tol=tol_lasso,max_iter=2000)
            model.fit(X_train,Y_train)

            ###Calculate test set MSE
            Y_hat =model.predict(X_test)
            n = len(Y_test)
            test_MSE = np.sum((Y_test-Y_hat)**2)**1/n
            test_MSE_array.append(test_MSE)
            if i > 0:
                conv_test = (np.average(test_MSE_array[:]) - np.average(test_MSE_array[:-1]))**2
                if conv_test < tol_boot:
                    break
            if i == MSSI:
                print("%s lambda value did not converge" % alpha_array[j])
        lambda_test_MSE_averages.append(np.average(test_MSE_array))
        if (j+1/len(alpha_array)*10)%10 == 0 & wrapper==False:
            print('Job %s' % str(j+1/len(alpha_array)*10), r'% complete' )
    optimum_lambda = alpha_array[lambda_test_MSE_averages.index(min(lambda_test_MSE_averages))]
    print("Job complete, optimum lambda converged on %s" % optimum_lambda)
    if display:
        displayResult(alpha_array, lambda_test_MSE_averages)
        displayFeatures(optimum_lambda, tol_lasso, X_train, Y_train, data, datadf)
    return lambda_test_MSE_averages, alpha_array


def displayFeatures(optimum_lambda, tol_lasso, X_train, Y_train, data, datadf):
    model = Lasso(alpha=optimum_lambda,tol=tol_lasso)
    model.fit(X_train,Y_train)
    i=0
    for a in range(len(data[0])-1):
        if model.coef_[a] != 0:
            print(a,datadf.columns[a])
            i+=1
    print("%s total features selected" % i)
    
def displayResult(alpha_array, averages, variances=None, error=False):
    optimum_alpha=alpha_array[np.argmin(np.absolute(averages))]
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(14,14))
        if error==False:
            plt.plot(np.log(alpha_array), averages)
        else:
            plt.errorbar(np.log(alpha_array), np.absolute(averages), np.absolute(variances))
            plt.plot(np.log(optimum_alpha), np.min(np.absolute(averages)), \
                c='r', marker='*', markersize=20, label=optimum_alpha)
        plt.legend()
        plt.ylabel('average MSE')
        plt.xlabel('lambda')
        plt.grid(False)
        plt.show()
        
def myround(x, base):
    return (float(base) * round(float(x)/float(base)))

def validationWrapper(name_of_pickle, iterations=30, TSF=0.8, BSF=0.8, alpha_array=np.arange(1,0,-1e-2),\
                    invisible_test=False, method="bootstrap"):
    """
    wrapper for boostrapLassoInvisibleTest. TSF is the test split fraction, the 
    test MSE dataset that the bootstrap will not sample from for a given iteration.
    
    example usage:
    
    """
    datadf = pd.read_pickle(name_of_pickle)
    data=np.array(datadf)
    results=np.zeros((len(alpha_array),iterations))
    for p in range(iterations):
        print("performing iteration %s of %s" % (p+1, iterations))
        if invisible_test:
            if method=="bootstrap":
                dataRand = np.random.permutation(data)
                n = data.shape[0]
                n_train = int(n*TSF) #set fraction of data to be for training
                result, returned_alpha_array = bootstrapLassoInvisibleTest(datadf, dataRand[:n_train,:],\
                        dataRand[n_train:,:], alpha_array=alpha_array, BSF=BSF, wrapper=True, TSF=TSF)
        else:
            if method=="bootstrap":
                print("running bootstrap")
                result, returned_alpha_array = bootstrapLasso(datadf,\
                        alpha_array=alpha_array, BSF=BSF, wrapper=True,\
                        display=False)
            elif method=="shuffleSplit":
                print("running shuffle split")
                result, returned_alpha_array = shuffleSplitLasso(datadf,\
                        alpha_array=alpha_array, SSF=BSF, wrapper=True,\
                        display=False) 
            elif method=="cv":
                print("running cross validation")
                result, returned_alpha_array = shuffleSplitLasso(datadf,\
                        alpha_array=alpha_array, wrapper=True,\
                        display=False) 
                
        results[:,p]=result
    avg = np.mean(results, axis=1)
    std = np.std(results,ddof=1,axis=1)
    return avg, std, results
