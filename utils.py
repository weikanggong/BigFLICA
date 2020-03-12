#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:35:34 2018

@author: wgong
"""

import numpy as np
from scipy import linalg

def nets_zscore(x):
    # x : a nsubject * nfeature numpy matrix
    
    x_zscore=(x-x.mean(axis=0))
    stds=x.std(axis=0)
    index=stds==0
    
    if sum(index)>0:
        stds[index]=0.1
        print('Warning: '+str(sum(index))+' of the features are all zero or constants')
        print('Normalizing them to all zeros ...')
    
    x_zscore=x_zscore/stds
    
    return x_zscore

def BWAS_correlation(fMRI_2D_1,fMRI_2D_2):
    # fMRI_2D_1 and fMRI_2D_2 are both time * voxel (t * p1 and t * p2) matrices
    # This function return the fisher z transformed correlation matrix (p1 * p2)
    
    fMRI_2D_1 = (fMRI_2D_1 - fMRI_2D_1.mean(axis=0)) / fMRI_2D_1.std(axis=0)
    fMRI_2D_2 = (fMRI_2D_2 - fMRI_2D_2.mean(axis=0)) / fMRI_2D_2.std(axis=0)
    r=np.dot(np.transpose(fMRI_2D_1),fMRI_2D_2) / fMRI_2D_1.shape[0]
    
    return r


def SingleModality_MIGP(x, k = 10 ,subj_batch = 200, n_epoch = 1):
    #x is a nsubject * nfeature matrix
    #Online PCA across the rows of x, to extract k PCs
    #output is a nfeature * k matrix
    
    for j in range(0,n_epoch):
        
        print('Epoch: ' + str(j+1)+'...')
        
        d1=x.shape[0]
        
        if j>=1:
            x = x[np.random.permutation(d1),:]
        
        subj_batch = subj_batch + k
        
        ind_end=int(d1/np.float64(subj_batch)-1e-4)+1
        
        if j==0:
            st=int(0*subj_batch)
            en=min(d1,int((0+1)*subj_batch))
            
            W=x[st:en,:]
    
        for i in range(1,ind_end):
            
            #print(i)
            st=int(i*subj_batch)
            en=min(d1,int((i+1)*subj_batch))
            
            
            W=np.vstack((W,x[st:en,:]))
            
            d,u=linalg.eigh(np.dot(W,W.T),eigvals=(W.shape[0]-2*k,W.shape[0]-1))
            
            d=np.real(d)
            u=np.real(u)
            indx1=np.argsort(-d)
            d=d[indx1]
            u=u[:,indx1]
            
            W=np.dot(u.T,W)


    us = W[0:k,:].T

    #u = us / np.sqrt(np.sum(us**2,axis = 0))
    
    return us


def MultiModality_MIGP(x, k = 10 ,subj_batch = 200, n_epoch = 1, zscore = False):
    #x is a list of length k , each is [nsubject * nfeature]
    #doing online pca on feature dimensions
    #equivalent to do a pca on feature concat data nsubject * (nfeature * k)
    #output a nsubject * k matrix
    
    nmod = len(x)
    for i in range(0,nmod):
        print('Multi-Modality MIGP for Modality '+ str(i+1)+'...')
        if i==0:
            if zscore==True:
                uu = SingleModality_MIGP(nets_zscore(x[i]).T, k  ,subj_batch , n_epoch )
            else:
                uu = SingleModality_MIGP(x[i].T, k  ,subj_batch , n_epoch )
        else:
            if zscore==True:
                uu = SingleModality_MIGP(np.hstack((uu,nets_zscore(x[i]))).T, k  ,subj_batch , n_epoch )
            else:
                uu = SingleModality_MIGP(np.hstack((uu,x[i])).T, k  ,subj_batch , n_epoch )

    u = uu[:,0:k]

    return u


def MultiModality_MIGP_faster(x, k = 10 ,subj_batch = 200, n_epoch = 1, zscore = True):
    #x is a list of length k , each is [nsubject * nfeature]
    #doing online pca on feature dimensions
    #equivalent to do a pca on feature concat data nsubject * (nfeature * k)
    #output a nsubject * k matrix
    
    
    nmod=len(x)
    nsub=x[0].shape[0]
    
    cov_mat=np.zeros((nsub,nsub))
    for i in range(0,nmod):
        dat=nets_zscore(x[i])
        cov_mat=cov_mat+np.dot(dat,dat.T)/dat.shape[1]
    
    if nsub<1200:
        dd,uu=linalg.eigh(cov_mat,eigvals=(nsub-k,nsub-1))
        dd=np.real(dd)
        uu=np.real(uu)
        indx1=np.argsort(-dd)
        dd=dd[indx1]
        uu=uu[:,indx1]
        uu = np.dot(uu,np.diag(dd))
    else:
        uu = SingleModality_MIGP(cov_mat, k  ,subj_batch , n_epoch )
    
    u = uu[:,0:k]
    
    return u




def nets_svds(x,nComp):
    # x : a nsubject * nfeature numpy matrix
    # nComp : the number of dimension (int), should be < min(x.shape[0], x.shape[1])
    
    if x.shape[0] < x.shape[1]:
        
        cov_mat=np.dot(x,x.T)
        if nComp < x.shape[0]:
            
            d,u=linalg.eigh(cov_mat,eigvals=(x.shape[0]-nComp,x.shape[0]-1))
            d=np.real(d)
            u=np.real(u)
            indx1=np.argsort(-d)
            d=d[indx1]
            u=u[:,indx1]
    
        s = np.sqrt(np.abs(d))
        v = np.dot(x.T , np.dot(u , np.diag(1/s)  ))

    else:
        
        cov_mat=np.dot(x.T,x)
        if nComp < x.shape[1]:
            
            d,v=linalg.eigh(cov_mat,eigvals=(x.shape[1]-nComp,x.shape[1]-1))
            d=np.real(d)
            v=np.real(v)
            indx1=np.argsort(-d)
            d=d[indx1]
            v=v[:,indx1]
            
            s = np.sqrt(np.abs(d));
            u = np.dot(x , np.dot(v , np.diag(1/s)  ))
        
    return u,s,v



def nets_dmean(x):
    # x : a nsubject * nfeature numpy matrix
    
    x_dmean=x-x.mean(axis=0)
    
    return x_dmean

def nets_melodic_normalization(x):
    # do fsl melodic-style data normalization across subjects
    # test: x=np.random.randn(100,1000)+1
    
    grot=nets_dmean(x)
    uu,ss,vv=nets_svds(grot,30)
    ss=np.diag(ss)
    vv[np.abs(vv)<2.3*np.std(vv)]=0     
    stddevs=np.std(grot-np.dot(np.dot(uu,ss),vv.T),axis=0)
    stddevs[stddevs<0.001]=0.001
                  
    grot=grot/stddevs  # var-norm
    
    x_normalized=nets_dmean(grot)
    
    return x_normalized
    
    
def sKPCR_regression(X,Y,cov):

    contrast=np.transpose(np.hstack(  ( np.eye(X.shape[1],X.shape[1]) , np.zeros((X.shape[1],cov.shape[1])) ))   )
    contrast=np.array(contrast,dtype='float32')
    
    design=np.hstack((X,cov))
    #degree of freedom
    df=design.shape[0]-design.shape[1]
    #
    ss=np.linalg.inv(np.dot(np.transpose(design),design))

    beta=np.dot(np.dot(ss,np.transpose(design)),Y)

    Res=Y-np.dot(design,beta)

    sigma=np.reshape(np.sqrt(np.divide(np.sum(np.square(Res),axis=0),df)),(1,beta.shape[1]))

    tmp1=np.dot(beta.T,contrast)
    tmp2=np.array(np.diag(np.dot(np.dot(contrast.T,ss),contrast)),ndmin=2)

    Tstat=np.divide(tmp1,np.dot(sigma.T,np.sqrt(tmp2)  ))


    return Tstat


def BWAS_deconf(X,Y):
    
    X=np.hstack((X,np.ones((X.shape[0],1))))
    
    ss=np.linalg.pinv(np.dot(np.transpose(X),X))
    
    beta=np.dot(np.dot(ss,np.transpose(X)),Y)
    
    Res=Y-np.dot(X,beta)
    
    return Res








