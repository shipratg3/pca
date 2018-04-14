# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 01:40:17 2018

@author: shipratg3
"""

def pca(data):
    """Plot error graph of PCA for a given data set data."""
    import numpy as np
    import matplotlib.pyplot as plt 
    data=data.select_dtypes([np.number])
    data.dropna(how="all", inplace=True) # drops the empty line at file-end
    #df=df.drop('class',1)
    data.shape
    mat_data=np.matrix(data)
    dataset=np.transpose(mat_data)
    for i in range(len(dataset)):#loop to normalize the data set
        mean=np.mean(dataset[i])
        std=np.std(dataset[i])
        dataset[i]=(dataset[i]-mean)/std
    cov=np.cov(dataset)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    mat=np.matrix(eig_vecs)
    err_list=[]
    for i in range(len(mat)):#loop to find sum square error
        PC=mat[:,0:i+1] # Principal Components
        f=np.matmul(PC.T,dataset)
        approx=np.matmul(PC,f)
        err_list.append(np.sum(np.square(dataset-approx))) #computing sum squared error
        
    #plotting the errors
    n=min(dataset.shape)+1
    plt.plot(range(1,n),err_list,color='red', marker='s',markerfacecolor='green', markersize=8)
    plt.title('PCA Vs SSE \n')
    plt.xlabel('Principal Component')
    plt.ylabel('Sum Square Error')
    plt.show()