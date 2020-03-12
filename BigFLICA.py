#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:44:27 2020

@author: wgong
"""

import os 
import numpy as np
from copy import deepcopy
import BigFLICA.FLICA_cpu as flica
import BigFLICA.utils as utils
import spams 
import scipy

def BigFLICA(data_loc, nlat, output_dir, migp_dim, dicl_dim):
    '''
    data_loc: a list with length equals to the number of modalities, each is the absolute directory of data
              matrix of one modality in .npy format. The data matrix is assumed to be subject * voxels.
              The number of subjects should be equal across modalities.
    nlat: Number of components to extract in BigFLICA
    output_dir: the absolute directory to store all BigFLICA results
    migp_dim: Number of components to extract in MIGP step.
    dicl_dim: Number of components to extract in the Dictionary learning step
    '''
    
    os.system('mkdir '+output_dir)
    os.system('mkdir '+output_dir + '/MIGP')
    os.system('mkdir '+output_dir + '/DicL')
    
    tmp = np.load(data_loc[0])
    nmod = len(data_loc)
    nsubj = tmp.shape[0]    
    
    print('MIGP...')
    cov_mat = np.zeros((nsubj,nsubj))
    for i in range(0,nmod):
            
        print('Loading data of Modal '+str(i+1))        
        Data = np.load(data_loc[i])
        print('zscore normalization...')    
        Data = utils.nets_zscore(Data)    
        print('Computing covariance matrix...')
        cov_mat = cov_mat + np.dot(Data,Data.T)/Data.shape[1]

    
    if nsubj<1200:
        dd,uu=scipy.linalg.eigh(cov_mat,eigvals=(nsubj-migp_dim,nsubj-1))
        dd=np.real(dd)
        uu=np.real(uu)
        indx1=np.argsort(-dd)
        dd=dd[indx1]
        uu=uu[:,indx1]
        U = np.dot(uu,np.diag(dd))
    else:
        U = utils.SingleModality_MIGP(cov_mat, migp_dim  ,200 , 5 )
            
    np.save(output_dir+'/MIGP/U_mMIGP.npy',U)
    
    
    for i in range(0,nmod):        
        print('Loading data of Modal '+str(i+1))        
        Data=np.load(data_loc[i])    
        print('zscore normalization...')    
        Data = utils.nets_zscore(Data)          
        data_to_use=np.dot(Data.T,U)
    
        np.save(output_dir+'/MIGP/PCAdata_mod'+('%02d' % (i+1))+'.npy',data_to_use)
    
    
    #DicL parameters
    print('DicL...')
    param = { 'K' : dicl_dim, # learns a dictionary with dicl_dim elements, you can change this
              'lambda1' : 1,'lambda2': 1 , 'numThreads' : 1, 'batchsize' : 512,
              'iter' : 50}
    
    print('Loading data ...')
    list_of_arrays=[np.array(a) for a in range (0,nmod)]
    Data = deepcopy(list_of_arrays)
    
    #one can parallize this step
    for i in range(0,nmod):       
        #load original data, a subject-by-feature matrix
        Data=np.load(output_dir+'/MIGP/PCAdata_mod'+('%02d' % (i+1))+'.npy') 
        D = spams.trainDL(np.asfortranarray(Data.T,dtype ='float64'),**param).T #this is component-by-subject    
        np.save(output_dir+'/DicL/DLdata_mod'+('%02d' % (i+1))+'.npy',D)
        
        Data[i] = D
    
    
    print('FLICA...')
    
    print('Priors ...')
    opts = {"num_components":100,"maxits":1000,'lambda_dims':'o'
            ,"initH":"PCA","dof_per_voxel":'auto_eigenspectrum','computeF':0,
            'output_dir':output_dir} 
    
    Priors, Posteriors, Constants=flica.flica_init_params(Data,opts)
        
    print('Performing Inference ...')
    Morig = flica.flica_iterate(Data,opts,Priors,Posteriors,Constants)
    
    
    
    
    
    
    
data_dir=os.path.join(args.data_dir,'')   
#data_dir='/gpfs2/well/win/users/eov316/ukb_flica/results_new/IC25/iter500/'


data_dir=os.path.join(data_dir,'')
X=[]
for i in range(0,47):
    X.append(np.load(data_dir+'flica_X'+str(i+1)+'.npy'))
M=np.load(data_dir+'flica_result.npz')    
   

K=len(X)
R=M['H'].shape[1]
for k in range(0,K):
    #M.X{k} * diag(M.W{k}.*sqrt( M.H.^2 * makesize(M.lambda{k},[R 1]) * M.DD(k))')]; %#ok<AGROW>
    if np.matrix(M['lambda1'][k]).shape[0]==R:
        tmp=np.dot(np.square(M['H']),M['lambda1'][k])
    else:
        tmp=np.dot(np.square(M['H']),tile(M['lambda1'][k],[R,1]))
    tmp2=np.sqrt(np.dot(tmp,M['DD'][k]))
    tmp3=np.diag(np.multiply(M['W'][k],tmp2))
    tmp4=np.dot(X[k],np.diag(tmp3))
    if k==0:
        Xcat=copy.deepcopy(tmp4)
    else:     
        Xcat=np.concatenate((Xcat,tmp4))
        
weight = np.sum(np.square(Xcat),0)      
order=np.argsort(weight)
order=order[::-1]
weight=weight[order]
#order[find(weight==0)]=[]

np.save(data_dir+'new_order.npy',order)
np.save(data_dir+'new_weight.npy',weight)    
    
    
    
    
    return Morig
