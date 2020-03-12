#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:17:23 2019

@author: wgong
"""

import numpy as np
import copy
import scipy as sc
from numpy.lib.stride_tricks import as_strided
from pylab import size #* #for the find command or size
from pylab import identity
from pylab import trace
from scipy import interpolate
from scipy.optimize import fmin
import time
import os


def flica_parseoptions(R, opts={"num_components":10,"maxits":2000,"dof_per_voxel":"auto_eigenspectrum",
                                "lambda_dims":"R","initH":"PCA",'fs_path':'/Applications/freesurfer'}):
#set default options if not provided and do some small checks
#to do: add here more controls and maybe getting info for later on saving the data in right format etc....
    
    # Check any options that need to refer to the data:
    
    #R = Y[0].shape[1]; #number of subjects
    if  opts['num_components'] > R/4:
		print 'Consider using more subjects??'#, opts['initH'],'check me'
        
    if type(opts['initH'])!=str:
	#import pdb; pdb.set_trace()
        if opts['initH'].shape != np.ones([opts['num_components'], R]).shape:
              print 'The shape of the given H matrix does not have the right dimensions'
              print 'Returning to fully unsupervised, ignoring your H ....'
              opts['initH']='PCA'
                            
    return opts

def logdet(M,ignorezeros):
    if ignorezeros=='chol': 
        ld = 2*np.sum(np.log(np.diag(np.linalg.cholesky(M)),dtype="float32")) 
    else:
       print 'not implemented, not used in .m?'
       
    return ld
      
def apply3_logdet(X,ignorezeros):
    out=np.zeros([1,1,X.shape[2]]).astype('float32')
    for i in range (0,X.shape[2]):	
        out[:,:,i]=logdet(X[:,:,i],ignorezeros)    
    #test=np.apply_along_axis(various.logdet, 2, X,ignorezeros)#, *args, **kwargs)
    return out   
   


def sum_dims(M,dims):
#% Sum a matrix in various dimensions
#% BUT be prepared for the fact that the input matrix might be smaller than
#% it should be.
#% e.g. M is a 5x4 matrix and N is a 5x1 matrix, but conceptually they're
#% both the same size...
#%   sum_dims(M,[5 0]) = sum(M,1)
#%   sum_dims(N,[5 0]) = 5*sum(N,1) = 5*N
#%   sum_dims(M,[0 4]) = sum(M,2)
#%   sum_dims(N,[0 4]) = sum(N,2)
#%   sum_dims(M,[5 4]) = sum(M(:))
#%   sum_dims(N,[5 4]) = sum(N)*5.
#% An error will result if there's a size mismatch, e.g. sum_dims(M,[6 0]).  
    for d in range (0,len(dims)):
        if dims[d]==0:
            1
        elif dims[d]==M.shape[d]:
            M=np.sum(M,d,dtype="float64").astype("float32")
            if dims[d]==1: # added to match sum from matlab
                M=np.expand_dims(M,axis=d)
        elif (dims[d]>0) & (M.shape[d]==1):
            #M = M*dims[d] its correct
            M = np.multiply(M,dims[d],dtype="float32")
        else:# dims[d]>1 & M.shape[d]>1:
            print "some error, check .m"
    return M[0,0]

def apply3_diag(X):
    out=np.zeros([X.shape[0],X.shape[0],X.shape[2]]).astype('float32');
    for i in range (0,X.shape[2]):	
		#out[:,:,i]=np.diagflat(X[:,:,i])
        out[:,:,i]=np.diagflat(X[:,:,i])
    return out

def apply3_diag2(X):
	out=np.zeros([X.shape[0],X.shape[2]]).astype('float32');
	for i in range (0,X.shape[2]):	
		#out[:,:,i]=np.diagflat(X[:,:,i])
		out[:,i]=np.diag(X[:,:,i])
	return out

def inv_prescale(inp):
	prescale = np.diag(np.power(np.diag(inp),np.float32(-.5)))
    #GWK: MODIFIED FROM inv to pinv
	out = np.dot( np.dot( prescale,  np.linalg.inv( np.dot(np.dot(prescale,inp),prescale)) )   ,prescale)
	return out

def apply3_inv_prescale(X):
	out=np.zeros([X.shape[0],X.shape[0],X.shape[2]]).astype('float32');
	for i in range (0,X.shape[2]):	
		out[:,:,i]=inv_prescale(X[:,:,i])

	return out

def rms(IN, dim, options):
	if dim==[]:  #I use only this case USED FLICA LOAD!!
		out = np.sqrt(np.sum(np.square(IN))/IN.size)# dumm = alb_various.rms(Y[k],0,[])
	else:
		out = np.sqrt(np.divide(np.sum(np.square(IN),dim),IN.shape[dim]))# dumm = alb_various.rms(Y[k],0,[])
	return out



def est_DOF_eigenspectrum(S):
	#if size(S) == len(S):
		#    % Good!
		#    assert(all(diff(S(isfinite(S)))<0))
	#elif isequal(S.shape, [len(S) len(S)]) && isequal(S,diag(diag(S))):
		#    S = diag(S);
	#elif len(S.shape) == 2:
	if S.shape[1]>S.shape[0]:
		print 'error --> Matrix is wider than it is tall -- eigenspectrum method won''t work!'
	V,D = np.linalg.eig(np.dot(np.transpose(S),S))
	idx = V.argsort()[::-1]  
	s2=np.sqrt(V[idx])
	#    s2 = flipud(sqrt(eig(S'*S)));
	#    %[u S v] = svd(S,'econ');
	#    %S = diag(S);
	#    %assertalmostequal(S, s2);
	S = s2;
	#else:
	#    [~,S,~] = svd(reshape(S,[],size(S,4)),'econ');
	#    S = diag(S);
	#end
 

	#    % Use new analytic method!
	#if all(isfinite(S)) && nargin==1:
	#%        S(1:ceil(end/3)) = nan;
	#%        S(end-3:end) = nan;
	keep = np.zeros(S.shape)
	idx=np.ceil((len(S)*.25)-1); keep[idx.astype(int)] =1
	idx=np.floor((len(S)*.75)-1); keep[idx.astype(int)] =1
	#        %S(floor([1:end*.25-1, end*.25+1:end*.75-1, end*.75+1:end])) = nan; % Christian's recommended method
	#        %S(end) = nan; % Especially important to mask out the smallest eigenvalue, and sometimes the floor above doesn't quite catch it.  (484 data set ok, #'cuz it's a multiple of 4).
	noKeep=np.where(keep == 0)[0]	
	S[noKeep] = np.nan;
	#        assert(sum(isfinite(S))==2)
	#elif all(isfinite(S)):
	#        assert(isequal(r,'all'))
	
	#print 'keep going here'	
	gam = fit_eigenspectrum(np.square(S));
	dof = len(S) / gam[0];
	#dof =1
	return dof


def fit_eigenspectrum(spec):#, gam)

	#if nargin==2:
	#   assert(numel(spec)==1)
	#    gam = specest(spec, gam);
	#else:

	#% I guess a maximum-likelihood fit would be good, because with have the 
	#% PDF...
	#% But would you do that while excluding the top N points???

	#% Matlab's fminsearch allows you to set TolX, but it is always an absolute
	#% tolerance rather than relative, so it really only works sensibly when the
	#% parameters all have roughly the same scaling.  As a result, we prescale
	#% the spectrum to have a scale around 1 (rather than the 10^7 that Wooly
	#% keeps feeding me).

	prescaleSpec = np.median(spec[np.isfinite(spec)]);
	spec = spec/prescaleSpec;
	gam = np.array([.1, 1]);
	#gam=[.1,1]
	#%exitflag = 0
	#%while exitflag == 0
	#%disp 'Nonlinear fit...'
	#%[gam junk exitflag] = fminsearch(@(gam) misfit(spec,gam), gam) %[0.5 spec(end/2)])
	#%end
	#print 'keep going here'
	#gam = fminsearch(@(gam) misfit(spec,gam), gam); 
	misfit(gam,spec)
	gam = fmin(misfit,gam,args=(spec,));#method='Nelder-Mead')
	gam = np.multiply(gam, np.array([1, prescaleSpec]))  # Put the scaling back
	#gam = gam .* [1 prescaleSpec]; # Put the scaling back
	return gam

def nurange(gam,stepSize):
	out=np.arange( np.square((1-np.sqrt(gam))), np.square((1+np.sqrt(gam))) ,stepSize)
	return out

def ifeta(gam,stepSize):
	cc=nurange(gam,stepSize)
	aa=np.divide(1,(2*gam*np.pi*cc))
	bb=np.sqrt( ( cc- np.ndarray.min(cc))*( np.ndarray.max(cc)-cc )   )

	out=aa*bb
	return out

def specest(speclen, gam):
	if len(gam)==2:
	    scale = gam[1]
	    gam = gam[0]
	else:
	    scale = 1	

	if gam<=0:# gam >= 1 | scale < 0 :
		est=np.inf
	elif gam >= 1:
		est=np.inf
	elif scale <0:
		est=np.inf
	else:
		#% Note: in Johnstone[2001], gam <- 1/gam
		stepSize = .001;
		##nurange =  @(gam) (1-sqrt(gam)).^2 : stepSize : (1+sqrt(gam)).^2;
		##ifeta =  @(gam) 1./(2*gam*pi*nurange(gam)).*sqrt( (nurange(gam)-min(nurange(gam))).*(max(nurange(gam))-nurange(gam)) );

		nu = nurange(gam,stepSize) #nurange(gam);
		tmp=ifeta(gam,stepSize)
		cif = np.cumsum( tmp ) *stepSize;

		xranges = np.linspace(0,1,speclen);
		cif[-1] = 1;
		#assert(all(isfinite([cif(:);nu(:);xrange(:)])))
		#est = interp1(cif, nu, xrange);
#print 'keep going here'
		est=interpolate.interp1d(cif,nu,kind='linear')(xranges)
		#%assert(rms(est-est2)/rms(est)<1e-3))
		idx = est.argsort()[::-1]
		#est = flipud(est(:)) * scale;
		est=est[idx]*scale
	return est


def misfit(gam,spec):
	est = specest(len(spec), gam);
	ssd = np.square(spec-est);
	ssd = np.sum(ssd[np.isfinite(spec)]);
	return ssd




###main update small functions########################################################################
def update_X_k(input_dict):
    #if i==0:
    precalc_YlambdaHTW_NxL = np.dot(input_dict['Y_k'],np.multiply( np.dot(np.array(input_dict['lambda_R_k']),input_dict['W_k']),np.transpose(input_dict['H'])))
    
    
    for i in range(input_dict['L']):# (0,input_dict['L']): 
        #tt=time.time()                  
        input_dict['X_k'][:,i]=0  
        input_dict['X2_k'][:,i]=np.nan 
        
        #%% Update P'(X_i|q_i)
        #print(precalc_YlambdaHTW_NxL[:,i].shape)
        #print(input_dict['X_k'].shape)
        #print(input_dict['WtW_k'][:,i].shape)
        #print(input_dict['HlambdaHt_k'][:,i].shape)
                    
        tmpM_N = precalc_YlambdaHTW_NxL[:,i] - np.dot(input_dict['X_k'], np.multiply(input_dict['WtW_k'][:,i], np.matrix(input_dict['HlambdaHt_k'][:,i]).T))
        #tt2=time.time() 
        tmpM_MxN =  np.add(tmpM_N,np.matrix(np.multiply(input_dict['beta_k'][:,i],input_dict['mu_k'][:,i])),order='F') 
        #print 'cost_1_4 = ' , time.time()-tt2        
        tmpL_M = np.multiply(input_dict['WtW_k'][i,i] , input_dict['HlambdaHt_k'][i,i]) + input_dict['beta_k'][:,i] 
        tmpVpost = np.divide(np.float64(1),tmpL_M)
        input_dict['Xq_var_k'][i,:] = copy.deepcopy(tmpVpost)#deep copy? In tmpLogQand xq2ki I use tmpVpost but its originaly Xq_var[k][i,:]
        Xqki = np.divide( tmpM_MxN, np.matrix(tmpL_M),dtype="float64")  # Xqki, Xqki_sq and tmpM_MxN are also large, vovelsx3          
        Xqki_sq=np.square(Xqki,order='F')                
        Xq2ki = np.add(Xqki_sq, tmpVpost, order='F') 
        #%% Update P'(q)  
        tmpLogQ = np.matrix( np.divide(np.add(np.log(tmpVpost,dtype="float64") , np.subtract(input_dict['beta_log_k'][:,i],
                               np.multiply(input_dict['beta_k'][:,i],input_dict['mu2_k'][:,i])),order='F') ,np.float64(2)) + 
                            np.squeeze(input_dict['pi_log_k'][:,i])) 
                                                         
        tmpLogQ = np.add(tmpLogQ,np.divide(Xqki_sq,np.matrix(np.multiply(np.float64(2),tmpVpost,order='F')),dtype='float64',order='F'),dtype='float64',order='F') #Xqki_sq is large  voxelsx3 so tmpLogQ also from here on                                 
 
        tmpLogQ = np.subtract(tmpLogQ,  np.amax(tmpLogQ,1),dtype='float64',order='F')

        qki = np.exp(tmpLogQ,dtype='float64',order='F')     
        
        qki = np.divide(qki, np.array(np.sum(qki,1,dtype="float64")),dtype='float64',order='F') #better sum float64       
        #print 'cost_2_4 = ' , time.time()-tt2         
        input_dict['sumN_Dq_k'][:,i] = np.multiply(input_dict['DD_k'] , np.matrix(np.sum(qki,0,dtype='float64')),dtype='float64')
        input_dict['sumN_DqXq_k'][:,i] = np.multiply(input_dict['DD_k'] , np.matrix(np.sum(np.multiply(qki,Xqki),0,dtype='float64')),dtype='float64') 
        input_dict['sumN_DqXq2_k'][:,i] = np.multiply(input_dict['DD_k'] , np.matrix(np.sum( np.multiply(qki,  Xq2ki,order='F' ),0,dtype='float64')).astype('float64'))          
        tmp_qlogq = np.multiply(qki,np.log(qki,dtype="float64"),order='F')
        tmp_qlogq[qki==0] = 0;  #% limit as q->0 of q*log(q) is 0.                
        input_dict['sumN_Dqlogq_k'][:,i] = np.multiply(input_dict['DD_k'] , np.matrix(np.sum(tmp_qlogq,0,dtype='float64')),
                                          order='F', dtype='float64')
        input_dict['X_k'][:,i] = np.squeeze(np.sum( np.multiply(Xqki, qki,order='K', dtype='float64'), 1))
        input_dict['X2_k'][:,i] = np.squeeze(np.sum( np.multiply(Xq2ki , qki,order='K', dtype='float64'), 1) )
        
    output_X_k_dict={'X_k':input_dict['X_k'],'X2_k':input_dict['X2_k'],'sumN_Dqlogq_k':input_dict['sumN_Dqlogq_k'],
                    'sumN_DqXq2_k':input_dict['sumN_DqXq2_k'],'sumN_DqXq_k':input_dict['sumN_DqXq_k'], 'sumN_Dq_k':input_dict['sumN_Dq_k'],
                    'Xq_var_k':input_dict['Xq_var_k']}
    return output_X_k_dict 




def update_mixmod(input_dict):

    for k in range(input_dict['K']):            
            input_dict['XtDX'][k] = np.dot(np.dot(np.transpose(input_dict['X'][k]), input_dict['X'][k]) , input_dict['DD'][k]) #% [LxL]
            np.fill_diagonal(input_dict['XtDX'][k],np.dot( np.sum( input_dict['X2'][k],0,dtype="float64") , input_dict['DD'][k]) ) # replace diagonal to include covariance
            #%% Update P'(pi_mean)
            input_dict['pi_weights'][k] = input_dict['prior_pi_weights'][k] + input_dict['sumN_Dq'][k] # [3xL]
            input_dict['pi_mean'][k] = np.divide(input_dict['pi_weights'][k], np.matrix(np.sum(input_dict['pi_weights'][k],0)) )
            input_dict['pi_log'][k] =np.subtract( sc.special.psi(input_dict['pi_weights'][k]) , np.matrix(sc.special.psi(np.sum(input_dict['pi_weights'][k],0))) ); #% [3xL]                
            #%% Update P'(beta)
            input_dict['beta_c'][k] = input_dict['prior_beta_c'][k] + (0.5*input_dict['sumN_Dq'][k]); #% [3xL]
            tmp = np.multiply(input_dict['sumN_Dq'][k] , input_dict['mu2'][k]) + input_dict['sumN_DqXq2'][k] - (2* np.multiply( input_dict['mu'][k] , input_dict['sumN_DqXq'][k])); #% [NxLx3]
            input_dict['beta_binv'][k] = np.float64(1)/input_dict['prior_beta_b'][k] + (tmp/np.float64(2)); #% [3xL]
            input_dict['beta'][k] = np.divide( input_dict['beta_c'][k] , input_dict['beta_binv'][k]); #% [3xL]
            input_dict['beta_log'][k] = sc.special.psi(input_dict['beta_c'][k]) - np.log(input_dict['beta_binv'][k])   #if ~(all(beta{k}(:) > 1e-10)), warning 'X getting awfully large', end %#ok<WNTAG>
            #%% Update P'(mu)
            tmp_L = (1./input_dict['prior_mu_var']) + np.multiply(input_dict['beta'][k] , input_dict['sumN_Dq'][k] );# % [3xL]
            tmp_M = np.divide( input_dict['prior_mu_mean'] , input_dict['prior_mu_var']) + np.multiply( input_dict['beta'][k] , input_dict['sumN_DqXq'][k]); #% [3xL]
            input_dict['mu'][k] = np.divide(tmp_M,tmp_L); #% [3xL]
            input_dict['mu_var'][k] = 1./tmp_L; #% [3xL]
            input_dict['mu2'][k] = np.square(input_dict['mu'][k]) + input_dict['mu_var'][k]
            
    output_mixmod_dict={'XtDX':input_dict['XtDX'],'pi_weights':input_dict['pi_weights'],'pi_log':input_dict['pi_log'],'pi_mean':input_dict['pi_mean'],
                    'beta_c':input_dict['beta_c'],'beta_binv':input_dict['beta_binv'],'beta':input_dict['beta'],'beta_log':input_dict['beta_log'],
                    'mu':input_dict['mu'],'mu_var':input_dict['mu_var'],'mu2':input_dict['mu2']}
                    
    return output_mixmod_dict 


def update_eta(input_dict):
    eta_binv =np.transpose(np.matrix( (np.float64(1)/input_dict['prior_eta_b']) + (input_dict['H2Gmat']/np.float64(2)) ).astype('float64')); 
    eta_c = input_dict['prior_eta_c'] + np.tile( (np.sum(input_dict['Gmat'],0)/2).astype('float64'),(input_dict['L'], 1)) ; 
    eta = np.divide(eta_c , eta_binv) 
    eta_log = sc.special.polygamma(0, eta_c) - np.log(eta_binv,dtype="float64") 
    output_eta_dict={'eta_binv':eta_binv,'eta_c':eta_c, 'eta':eta, 'eta_log':eta_log}
    return output_eta_dict

def update_H(input_dict):        
     if 'R' in input_dict['opts']['lambda_dims']: 
         aaa=np.expand_dims(np.dot(input_dict['eta'],np.matrix(input_dict['Gmat'])),axis=1) #[L=NumIcas 1 R=NumSubs]
         tmpVinv_LxLxNH = apply3_diag(aaa);
         tmp_R_to_NH = range(0,input_dict['R']) #1:R;
     else: 
        aaa=np.transpose( np.expand_dims(input_dict['eta'],axis=2),(0 ,2 , 1))
        tmpVinv_LxLxNH = np.squeeze(apply3_diag(aaa)) 
        tmp_R_to_NH = np.ones(input_dict['R']).astype(int)      
     
     tmpM = np.zeros([input_dict['L'],input_dict['R']]).astype('float64') 
     alb_dum=np.empty([input_dict['K'],input_dict['L']]).astype('float64')
     alb_dum[:]=np.nan;
     input_dict['H_PCs'] = np.vstack([alb_dum, np.transpose(input_dict['eta'])]) 
     for k in range (0,input_dict['K']): 
         input_dict['W'][k]=np.squeeze(np.array(input_dict['W'][k].T)).astype('float64')
         tmp_lambda_NH = input_dict['Lambda'][k] + np.zeros([input_dict['NH'],1],dtype='float64'); 
         aaa=np.array(np.dot( np.multiply(input_dict['WtW'][k].flatten(1), input_dict['XtDX'][k].flatten(1)).T, tmp_lambda_NH.T))#,order="F")
         aaa=as_strided(aaa,shape=(input_dict['L'],input_dict['L'],input_dict['NH']))
         if input_dict['NH'] ==1:
             aaa=aaa[:,:,0] #try to remove this loop 
         tmpVinv_LxLxNH = tmpVinv_LxLxNH + aaa; #del aaa 
         spm=np.dot(np.transpose(input_dict['X'][k]), input_dict['Y'][k])
         tmpM = tmpM + input_dict['DD'][k] * np.dot( np.dot( np.diag(input_dict['W'][k]) , spm) , np.diag(np.array(input_dict['lambda_R'][k].flatten())[0])  )
         input_dict['H_PCs'][k] = np.dot( np.multiply(np.diag(input_dict['WtW'][k]) , np.diag(input_dict['XtDX'][k]) )  , np.mean(input_dict['lambda_R'][k],dtype='float64'))                                  

     # Calculate H, H covariance, <H*Ht> and <H*lambda*Ht>         
     if input_dict['NH']==1:
        input_dict['H_colcov'] = inv_prescale(tmpVinv_LxLxNH)
        input_dict['H'] = np.dot(input_dict['H_colcov'], tmpM);
        alb_dum3=np.diag(input_dict['H_colcov']) 
     else: 
        input_dict['H_colcov'] =  apply3_inv_prescale(tmpVinv_LxLxNH)
        for rr in range (0,input_dict['R']):
            input_dict['H'][:,rr] = np.dot(input_dict['H_colcov'][:,:,tmp_R_to_NH[rr]] , tmpM[:,rr])     
        alb_dum3=apply3_diag2(input_dict['H_colcov']) 
        
     if input_dict['NH']==input_dict['R']:
        input_dict['H2Gmat'] = np.dot( np.square(input_dict['H']) ,input_dict['Gmat']) + np.dot(alb_dum3 ,input_dict['Gmat']) # [LxG]
     else: #not tested
        input_dict['H2Gmat'] = np.dot(np.square(input_dict['H']) , input_dict['Gmat']) + np.dot(alb_dum3 , np.dot( np.transpose(input_dict['Gmat']),input_dict['Gmat'])) # [LxG] 
        
     output_H_dict={'H':input_dict['H'],'H2Gmat':input_dict['H2Gmat'], 'H_colcov':input_dict['H_colcov'], 
                    'H_PCs':input_dict['H_PCs'], 'tmp_R_to_NH':tmp_R_to_NH, 'W':input_dict['W']}
     
     return output_H_dict    
 

def update_HlambdaHt_and_W(input_dict):            
    for k in range (0,input_dict['K']):
            input_dict['HlambdaHt'][k] = np.dot( np.dot(input_dict['H'] , np.diag(np.array(input_dict['lambda_R'][k].flatten())[0])    ) ,np.transpose(input_dict['H']))
            if size(input_dict['H_colcov'].shape)==2:
                ss=np.dot( np.transpose(input_dict['Gmat']) , input_dict['lambda_R'][k])
                input_dict['HlambdaHt'][k] = np.add(input_dict['HlambdaHt'][k] , np.multiply(input_dict['H_colcov'] , ss[0,0]) )  
            else:   
                if input_dict['H_colcov'].shape[2] == input_dict['R']:
                    for rr in range(0,input_dict['R']):
                        input_dict['HlambdaHt'][k] = np.add( input_dict['HlambdaHt'][k] , np.multiply(input_dict['H_colcov'][:,:,rr] , np.array(input_dict['lambda_R'][k][rr])) )
                else: # size(H_colcov,3) == G
                    for g in range (0,1):#G):
                        input_dict['HlambdaHt'][k] = input_dict['HlambdaHt'][k] + np.dot(input_dict['H_colcov'][:,:,g] , np.dot( np.transpose(input_dict['Gmat'][:,g]) , input_dict['lambda_R'][k])) 
                        
            #%% Update W 
            tmpL = np.multiply(input_dict['XtDX'][k] , input_dict['HlambdaHt'][k]) + ( (1./input_dict['prior_W_var']) * identity(input_dict['L'])).astype('float64') 
            tmpCov = inv_prescale(tmpL);
            input_dict['W_rowcov'][k] = (np.float64(0.5)*(tmpCov+np.transpose(tmpCov))).astype('float64') 
            spm=np.dot( np.transpose(input_dict['X'][k]) , input_dict['Y'][k]) 
            tmpM = np.diag(np.dot( np.dot( spm , np.diag(np.array(input_dict['lambda_R'][k].flatten())[0]) ) , np.transpose(input_dict['H']) )) * input_dict['DD'][k]
            input_dict['W'][k] = np.dot(np.matrix(tmpM) , input_dict['W_rowcov'][k])
            input_dict['WtW'][k] = np.matrix(np.dot(np.transpose(input_dict['W'][k]), input_dict['W'][k]) + input_dict['W_rowcov'][k]) 
                        
    output_HlamW_dict={'HlambdaHt':input_dict['HlambdaHt'],'W':input_dict['W'], 'WtW':input_dict['WtW'], 'W_rowcov':input_dict['W_rowcov']}
    return output_HlamW_dict 


def update_lambda(input_dict):
    
     for k in range (0,input_dict['K']):   
            tmp_diagHtWXtDXWH = np.sum( np.multiply(input_dict['H'] , np.dot(np.multiply(input_dict['WtW'][k],input_dict['XtDX'][k]),input_dict['H']))  , 0 )
            if size(input_dict['H_colcov'].shape)==2:
                tmp_diagHtWXtDXWH = tmp_diagHtWXtDXWH + np.dot( np.dot( np.multiply(input_dict['WtW'][k].flatten(1) ,np.matrix(input_dict['XtDX'][k].flatten(1))) , input_dict['H_colcov'].reshape(input_dict['L']*input_dict['L'],1
,order='F')), np.matrix(input_dict['Gmat']));
                                                                       
                                                                       
            else:
                if input_dict['H_colcov'].shape[2]==input_dict['R']:
                    tmp_diagHtWXtDXWH = tmp_diagHtWXtDXWH + np.dot( np.multiply(input_dict['WtW'][k].flatten(1) ,np.matrix(input_dict['XtDX'][k].flatten(1))) , input_dict['H_colcov'].reshape(input_dict['L']*input_dict['L'],input_dict['H_colcov'].shape[2]
,order='F'));
                else: # not tested !!!!!!!!!!!!!!!!!!!!!!
                    tmp_diagHtWXtDXWH = tmp_diagHtWXtDXWH + np.dot( np.dot( np.multiply(input_dict['WtW'][k].flatten(1) ,np.matrix(input_dict['XtDX'][k].flatten(1))) , input_dict['H_colcov'].reshape(input_dict['L']*input_dict['L'],input_dict['H_colcov'].shape[2]
,order='F')), np.transpose(input_dict['Gmat']));
            
            input_dict['lambda_c'][k] = (input_dict['DD'][k]*input_dict['N'][k]/2) * np.ones([input_dict['R'],1]); #% [Rx1]
            input_dict['lambda_binv'][k] = ((0.5*input_dict['DD'][k]* np.matrix(np.sum(np.square(input_dict['Y'][k]),0)) ) - (np.dot(np.multiply(np.dot(np.transpose(np.matrix(input_dict['X'][k])), input_dict['Y'][k]) ,input_dict['H']).T , input_dict['W'][k].flatten(1).T) * input_dict['DD'][k]).T + (0.5*tmp_diagHtWXtDXWH) ).T #% [Rx1]

            if input_dict['opts']['lambda_dims'] == 'R':
                1  #% OK!  lambda_c and lambda_binv are already Rx1
            elif input_dict['opts']['lambda_dims'] == 'G': # alb--> option G tested in python?
                input_dict['lambda_c'][k] = np.dot( np.matrix(input_dict['Gmat']).T , input_dict['lambda_c'][k]);
                input_dict['lambda_binv'][k] = np.dot( np.transpose(input_dict['Gmat']) , input_dict['lambda_binv'][k]);
            elif input_dict['opts']['lambda_dims'] == 'o':
                input_dict['lambda_c'][k] = np.sum(input_dict['lambda_c'][k]);
                input_dict['lambda_binv'][k] = np.sum(input_dict['lambda_binv'][k]);
            else:
                print 'Unimpleneted'
                
                
            input_dict['lambda_c'][k] = input_dict['lambda_c'][k] + input_dict['prior_lambda_c'][k];
            input_dict['lambda_binv'][k] = input_dict['lambda_binv'][k] + (1./input_dict['prior_lambda_b'][k]);
            input_dict['Lambda'][k] = np.divide(input_dict['lambda_c'][k] , input_dict['lambda_binv'][k] ); #% [Rx1 or Gx1 or 1x1]#assert(all(lambda{k}>0))
            input_dict['lambda_log'][k] = sc.special.psi(input_dict['lambda_c'][k]) - np.log(input_dict['lambda_binv'][k])
            
            
            if input_dict['opts']['lambda_dims'] == 'R':
                input_dict['lambda_R'][k] = input_dict['Lambda'][k] + np.zeros([input_dict['R'],1]);
                input_dict['lambda_log_R'][k] = input_dict['lambda_log'][k] + np.zeros([input_dict['R'],1]);
            elif input_dict['opts']['lambda_dims'] == 'G': # alb--> G option not tested in python?
                input_dict['lambda_R'][k] = np.dot(input_dict['Gmat'] , input_dict['Lambda'][k]);
                input_dict['lambda_log_R'][k] = np.dot(input_dict['Gmat'] , input_dict['lambda_log'][k]);
            elif input_dict['opts']['lambda_dims'] == 'o': # same as case 'R"
                input_dict['lambda_R'][k] = np.matrix(input_dict['Lambda'][k] + np.zeros([input_dict['R'],1]));
                input_dict['lambda_log_R'][k] = np.matrix(input_dict['lambda_log'][k] + np.zeros([input_dict['R'],1]))
            else:
                print 'Unimpleneted'
            #%% Calculate <H*lambda{k}*H'>
            input_dict['HlambdaHt'][k] = np.dot( np.dot(input_dict['H'], np.diag(np.array(input_dict['lambda_R'][k].flatten())[0])) , input_dict['H'].T)
            if size(input_dict['H_colcov'].shape)==2:
                        sss=np.dot(np.matrix(input_dict['Gmat']),input_dict['lambda_R'][k])
                        input_dict['HlambdaHt'][k] = input_dict['HlambdaHt'][k] + ( input_dict['H_colcov'] * sss[0,0]  )                
            else:
                if input_dict['H_colcov'].shape[2] == input_dict['R']:
                    for r in range (0,input_dict['R']):
                        input_dict['HlambdaHt'][k] = input_dict['HlambdaHt'][k] + (input_dict['H_colcov'][:,:,r] * np.array(input_dict['lambda_R'][k][r]))
                else:# alb--> next options are not tested in python
                    for g in range (0,1):#G):
                        input_dict['HlambdaHt'][k] = input_dict['HlambdaHt'][k] + np.dot( input_dict['H_colcov'][:,:,g], np.dot(np.transpose(np.matrix(input_dict['Gmat'][:,g])),input_dict['lambda_R'][k])) 
                        
     output_Lambda_dict={'Lambda':input_dict['Lambda'],'HlambdaHt':input_dict['HlambdaHt'],'lambda_log_R':input_dict['lambda_log_R'], 'lambda_R':input_dict['lambda_R'],
                         'lambda_log':input_dict['lambda_log'], 'lambda_binv':input_dict['lambda_binv'],'lambda_c':input_dict['lambda_c']}       
     
     return output_Lambda_dict 
    
def compute_F(input_dict): #NEED TO IMPROVE SUM_DIMS ...
              
    for key,val in input_dict.items(): #load all
             exec(key + '=val')
             
    F = np.nan;  
    Fpart["Hprior"]=(sum_dims(np.dot(eta_log,np.matrix(Gmat)),[L, R])/2)- (np.log(2*np.pi)*L*R/2)- (sum_dims(np.multiply(eta,np.matrix(H2Gmat).T),[L,G])/2) 
    if size(H_colcov.shape)==2: #case lambda='o' 
        tmp1=logdet(H_colcov,'chol')
        Fpart["Hpost"] = 0.5*L*R*(1+2*np.pi) + 0.5*np.sum(Gmat)*tmp1;
    else: #case lambda='R' 
        tmp1= apply3_logdet(H_colcov,'chol')
        Fpart["Hpost"] = 0.5*L*R*(1+2*np.pi) + 0.5* sum_dims(tmp1,[1, 1, R])
    Fpart["etaPrior"] = -sum_dims(np.matrix(sc.special.gammaln(prior_eta_c)),[L, G]) +sum_dims(np.matrix(np.multiply(prior_eta_c-1,eta_log)),[L, G]) -sum_dims(np.matrix(prior_eta_c*np.log(prior_eta_b)),[L, G])  -sum_dims(np.matrix(eta/prior_eta_b),[L, G]);
    Fpart["etaPost"] = sum_dims(np.matrix(sc.special.gammaln(eta_c)),[L, G]) -sum_dims(np.multiply((eta_c-1),eta_log),[L, G])  +sum_dims(np.multiply(-eta_c,np.log(eta_binv)),[L, G]) +sum_dims(np.multiply(eta,eta_binv),[L, G]);            
    for kk in range(0,K):
        Fpart["Wprior"][kk] = sum_dims(np.matrix(np.log(1./prior_W_var,dtype="float64"),dtype="float64"),[1, L])/2 - np.log(2*np.pi,dtype="float64")*1*L/2 - trace(WtW[kk])/2/prior_W_var;
        Fpart["Wpost"][kk] = 0.5*1*L*(1+2*np.pi) + 0.5*logdet(W_rowcov[kk],'chol');
        Fpart["muPrior"][kk] = -0.5/prior_mu_var*sum_dims(np.matrix(mu2[kk]),[3, L])  +0.5*np.log(2*np.pi*prior_mu_var,dtype="float64") * 3*L;
        Fpart["muPost"][kk] = 0.5*(1+np.log(2*np.pi,dtype="float64"))*3*L +0.5*sum_dims(np.matrix(np.log(mu_var[kk],dtype="float64")),[3, L]);
        Fpart["betaPrior"][kk] = -np.mean(np.mean(sc.special.gammaln(prior_beta_c[kk])))*3*L +np.mean(np.mean( np.multiply( (prior_beta_c[kk]-1) , beta_log[kk])))*3*L -  np.mean(np.mean(  np.multiply(prior_beta_c[kk],np.log(prior_beta_b[kk]))))*3*L -np.mean(np.mean( np.multiply( 1./prior_beta_b[kk], beta[kk])))*3*L;
        Fpart["betaPost"][kk] = sum_dims(np.matrix(sc.special.gammaln(beta_c[kk])),[3, L]) -sum_dims(  np.matrix(np.multiply((beta_c[kk]-1),beta_log[kk])),[3, L]) +sum_dims(  np.matrix(np.multiply(beta_c[kk],-np.log(beta_binv[kk]))),[3, L]) +sum_dims(  np.matrix(np.multiply(beta_binv[kk],beta[kk])),[3, L]);            
        Fpart["piPrior"][kk] = sum_dims(np.matrix( sc.special.gammaln( sum_dims(np.matrix(prior_pi_weights[kk]),[3, 0]) )), [1, L] )  -sum_dims( np.matrix(sc.special.gammaln( prior_pi_weights[kk] )), [3, L]) +sum_dims( np.multiply( (prior_pi_weights[kk]-1) , pi_log[kk]), [3, L]);
        Fpart["piPost"][kk] = -sum_dims( np.matrix(sc.special.gammaln( sum_dims(np.matrix(pi_weights[kk]),[3, 0]) )), [1, L]) +sum_dims( np.matrix(sc.special.gammaln( pi_weights[kk] )), [3, L])  -sum_dims( np.matrix(np.multiply( (pi_weights[kk]-1) , pi_log[kk])), [3, L]);
        Fpart["qPrior"][kk] = sum_dims( np.matrix(np.multiply(sumN_Dq[kk] , pi_log[kk])), [3, L]);
        Fpart["qPost"][kk] = - sum_dims(np.matrix(sumN_Dqlogq[kk]), [3, L]);
        Fpart["Ylike1"][kk] = N[kk]*DD[kk]/2 * sum_dims(lambda_log_R[kk]-np.log(2*np.pi,dtype="float64"),[R, 1]);
        Fpart["Ylike2"][kk] = -0.5*Y2D_sumN[kk]*lambda_R[kk];
        Fpart["Ylike3"][kk] = DD[kk] * (np.dot( np.sum( np.multiply(Y[kk]  , np.dot(np.dot(X[kk],np.diagflat(W[kk])),H)),0) ,lambda_R[kk]));
        Fpart["Ylike4"][kk] = -0.5 * sum_dims(np.matrix( np.multiply(np.multiply( XtDX[kk] , HlambdaHt[kk]) , WtW[kk])), [L, L]);
        Fpart["lambdaPrior"][kk] = -np.sum(sc.special.gammaln(prior_lambda_c[kk])) +np.sum(np.multiply((prior_lambda_c[kk]-1),lambda_log[kk])) -np.sum(np.multiply(prior_lambda_c[kk],np.log(prior_lambda_b[kk]))) -np.sum(1./np.multiply(prior_lambda_b[kk],Lambda[kk]));
        Fpart["lambdaPost"][kk] = np.sum(sc.special.gammaln(lambda_c[kk])) -np.sum(np.multiply((lambda_c[kk]-1),lambda_log[kk])) -np.sum(np.multiply(lambda_c[kk],np.log(lambda_binv[kk]))) +np.sum(np.multiply(lambda_binv[kk],Lambda[kk]));
        Fpart["XPrior"][kk] = sum_dims( np.matrix((0.5 * np.multiply( (beta_log[kk]-np.log(2*np.pi,dtype="float64")) , sumN_Dq[kk])) - (0.5 * np.multiply(beta[kk] , sumN_DqXq2[kk])) + np.multiply( np.multiply(beta[kk] , mu[kk]) , sumN_DqXq[kk])  - (0.5* np.multiply( np.multiply( beta[kk] , mu2[kk]) , sumN_Dq[kk]))) , [3, L]);
        Fpart["XPost"][kk] = -sum_dims(np.matrix( -0.5*  np.multiply( sumN_Dq[kk], (1+np.log(2*np.pi,dtype="float64")+np.log(Xq_var[kk],dtype="float64")).T)), [3, L])
    F = np.sum(np.sum(Fpart.values()),dtype="float64") #np.sum(sum([i for i in Fpart.values()])) #sum_carefully(Fpart); % add up all the bits 
    return F, Fpart

def zeros32(*args, **kwargs):
    kwargs.setdefault("dtype", np.float64)
    return np.zeros(*args, **kwargs)
    


def flica_init_params(Y,opts):
    
    K = len(Y)   #num kinds of data 
    L = np.int(opts['num_components'])    #num_components;
    R = Y[0].shape[1]    #num of subjects 
    default_list_of_arrays=[np.array(a).astype('float64') for a in range (0,K)] #list to save variables
    
    #set default options if not provided    
    opts=flica_parseoptions(R, opts) 
    
    #Compute degrees of freedom per voxel, if not provided
    if opts['dof_per_voxel']=='auto_eigenspectrum':
        opts['dof_per_voxel'] = np.ones(K);
        for k in range (0,K): 
            if Y[k].shape[1]<Y[k].shape[0]:
                opts['dof_per_voxel'][k] = est_DOF_eigenspectrum(Y[k]) / (Y[k].shape[0])  # check alb.est_DOF_eigenspectrum ??             
            else:
                opts['dof_per_voxel'][k] =1.0
    DD = np.real(opts['dof_per_voxel'])


    # Multiply data by Virtual Decimation factor (often sqrt'd!) and Initialize <X> and <H> using PCA:
    N=np.zeros(K).astype('float64') #num of voxels per data type

    if opts['initH']=='PCA': 
        print('Initialize FLICA using concatenated PCA across modalities...')

        cov_mat=np.zeros((R,R))
        X=[np.array(a).astype('float64') for a in range (0,K)] #list to save variables    
        for k in range(0,K):
            Y[k]=np.ascontiguousarray(Y[k],dtype='float64')
            N[k] = Y[k].shape[0]
            cov_mat=cov_mat+np.dot(Y[k].T * np.sqrt(DD[k]),Y[k]* np.sqrt(DD[k]))
        ds,us=np.linalg.eig(cov_mat)
        us=np.real(us)
        ds=np.real(ds)
        indx1=np.argsort(-ds)
        ds=ds[indx1]
        us=us[:,indx1]
        
        H=np.dot(np.diag(ds[0:L]),us[:,0:L].T)
        for k in range (0,K):
            print(k+1)
            X[k]=np.dot(np.dot(np.linalg.pinv(np.dot(H,H.T)),H),Y[k].T * np.sqrt(DD[k])).T
            #X[k]=np.dot(Y[k] * np.sqrt(DD[k]),H.T) 
            
    if opts['initH']=='Bigdata': 
        
        print('Initialize FLICA using provided subject mode...')
        X=[np.array(a).astype('float64') for a in range (0,K)] #list to save variables    
        ##opts['U'] is a component * subject numpy matrix
        ##the rows of opts['U'] should be orthogonal
        H = np.divide(opts['U'] , K*np.sqrt(np.mean(DD)))
        for k in range (0,K):   
            print(k+1)
            Y[k]=np.ascontiguousarray(Y[k],dtype='float64')
            N[k] = Y[k].shape[0]
            X[k]=np.dot(Y[k] * np.sqrt(DD[k]),H.T)   
             
    if opts['initH']=='PCAnew': 
        
        print('Initialize FLICA using modality-wise PCA...')
        
        X=[np.array(a).astype('float64') for a in range (0,K)] #list to save variables    
        tmpV=np.zeros((R,L))
        for k in range(0,K):
            print(k+1)
            Y[k]=np.ascontiguousarray(Y[k],dtype='float64')
            N[k] = Y[k].shape[0]            
            [tmpU1,tmpS1,tmpV1]=np.linalg.svd(Y[k] * np.sqrt(DD[k]),full_matrices=False);
            tmpV1=np.transpose(tmpV1)
            tmpS1=np.diag(tmpS1)
            tmpU1 = np.dot(tmpU1[:,0:L],tmpS1[0:L, 0:L])
            tmpV1 = np.divide(tmpV1[:,0:L], 1./rms(tmpU1,0,[]));
            tmpU1 = np.divide(tmpU1, rms(tmpU1,0,[]));
            
            X[k]=tmpU1
            tmpV=tmpV+tmpV1
            
        #tmpU=np.vstack(tmpU)
        #may be we can use glm to get tmpV too!!
        #tmpV=tmpV/K
        H = np.divide(tmpV.T , K*np.sqrt(np.mean(DD)))
        
#    else:
#        tmpV = opts['initH'].T
#        tmpU = np.squeeze(np.linalg.lstsq(tmpV,tmpYcat.T)[0]).T 
 
        

    
    #I FIX G TO BE ONE, Rgroups>1 not implemented yet
    G=np.ones(1,dtype='int') 
    Gmat =np.ones(R).astype('float64')
    H2Gmat = np.dot( np.square(H) , Gmat.T) # [LxG]
    H_colcov =np.dot(np.matlib.eye(L),pow(10,-12)).astype('float64')
    
    #define variables    
    #X=copy.deepcopy(default_list_of_arrays)
    W=copy.deepcopy(default_list_of_arrays)
    W_rowcov=copy.deepcopy(default_list_of_arrays)
    WtW=copy.deepcopy(default_list_of_arrays)
    XtDX=copy.deepcopy(default_list_of_arrays)
    Y2D_sumN=copy.deepcopy(default_list_of_arrays)
    X2=copy.deepcopy(default_list_of_arrays)
    for  k in range (0,K): # De-concatenate to get X[k] estimates:
        #if k==0:
        #    X[k] = tmpU[0:N.astype(int)[0],0:L]; # / sqrt(DD(k));
        #else:
        #    X[k] = tmpU[ np.sum(N.astype(int)[0:k])  : np.sum(N.astype(int)[0:k+1])  , 0:L]         
        W[k] = np.multiply(np.ones(L).astype('float64') , np.sqrt(np.divide(np.mean(DD) ,DD[k])))  # so Y = X*diag(W)*H + noise;
        W_rowcov[k] = np.multiply(np.matlib.eye(L),pow(10,-12)).astype('float64') 
        prior_W_var = np.divide(np.ones(1).astype('float64'),DD[k])  
        WtW[k] = np.multiply(W[k][np.newaxis, :].T , W[k]) + W_rowcov[k];
        XtDX[k] = np.dot (np.dot(X[k].T , X[k]), DD[k]) # double prec.?
        Y2D_sumN[k] = np.multiply(DD[k] , np.sum(np.square(Y[k]),0) )    # double prec.?
        X2[k] = np.square(X[k])
    
    #Set up the models for P(X|params) and P(lambda):
    # define variables
    prior_pi_weights=copy.deepcopy(default_list_of_arrays)
    pi_weights=copy.deepcopy(default_list_of_arrays)
    pi_mean=copy.deepcopy(default_list_of_arrays)
    pi_log=copy.deepcopy(default_list_of_arrays)
    prior_beta_b=copy.deepcopy(default_list_of_arrays)
    prior_beta_c=copy.deepcopy(default_list_of_arrays)
    beta=copy.deepcopy(default_list_of_arrays)
    beta_log=copy.deepcopy(default_list_of_arrays)
    beta_c=copy.deepcopy(default_list_of_arrays)
    beta_binv=copy.deepcopy(default_list_of_arrays)
    mu = copy.deepcopy(default_list_of_arrays)
    mu2 =copy.deepcopy(default_list_of_arrays)
    mu_var =copy.deepcopy(default_list_of_arrays)    
    prior_lambda_b=copy.deepcopy(default_list_of_arrays)
    prior_lambda_c=copy.deepcopy(default_list_of_arrays)
    Lambda=copy.deepcopy(default_list_of_arrays) # I use capital in lambda from .m!!
    lambda_log=copy.deepcopy(default_list_of_arrays)
    lambda_c=copy.deepcopy(default_list_of_arrays)
    lambda_binv=copy.deepcopy(default_list_of_arrays)
    lambda_R=copy.deepcopy(default_list_of_arrays)
    lambda_log_R=copy.deepcopy(default_list_of_arrays)
    HlambdaHt=copy.deepcopy(default_list_of_arrays)
    sumN_Dq=np.ndarray([K,3,L]).astype('float64')
    sumN_DqXq=np.ndarray([K,3,L]).astype('float64')
    sumN_DqXq2=np.ndarray([K,3,L]).astype('float64')
    sumN_Dqlogq=np.ndarray([K,3,L]).astype('float64')
    qq=[np.ndarray(a).astype('float64') for a in range (0,K)]
    Xq_var=[np.ndarray(a).astype('float64') for a in range (0,K)]
    for  k in range (0,K):
        #Initialize pi_mean{k} [3xL]
        prior_pi_weights[k] = (N[k]*0.1 * np.ones([3, L])).astype('float64'); 
        pi_weights[k] = copy.deepcopy(prior_pi_weights[k]);
        dumm=np.divide( pi_weights[k], np.tile(np.sum(pi_weights[k],0),(pi_weights[k].shape[0], 1)) )
        pi_mean[k] = copy.deepcopy(dumm)
        pi_log[k] = np.log(pi_mean[k])
        # Initialize beta{k} [3xL]
        prior_beta_b[k] = np.tile([pow(10,3), 1, pow(10,3)],(L,1)).T.astype('float64')
        prior_beta_c[k] = np.tile([pow(10,-6), pow(10,6), pow(10,-6)],(L,1)).T.astype('float64') 
        beta[k] = np.tile(np.power([.1, 1000., 1.],-2), (L,1)).T.astype('float64')  #TEST??
        beta_log[k] = np.log(beta[k])        
        beta_c[k] = np.multiply(np.power(10,6),np.ones(beta[k].shape)).astype('float64')        
        beta_binv[k] = np.divide(np.power(10,6),beta[k]).astype('float64');
        #Initialize mu{k} [3xL]:
        prior_mu_mean = np.zeros(1).astype('float64') 
        prior_mu_var = pow(10,4)*np.ones(1).astype('float64') 
        mu[k] = prior_mu_mean + np.zeros([3,L]).astype('float64')
        mu2[k] = np.square(mu[k]) 
        mu_var[k] = (np.multiply(mu[k], 0)+pow(10,-12)).astype('float64') 
        #Initialize q{k} [NxLx3]:
        qq[k] = np.tile(pi_mean[k].T, (N[k].astype('int'), 1, 1))
        #Initialize X_q [NxLx3]:
        Xq_var[k] = np.multiply(pow(10,-12), np.ones([L,3])).astype('float64') #######################################################
        #Set up the model for lambda: 
        if opts['lambda_dims'] == 'R':
            prior_lambda_b[k] = np.multiply( pow(10,12), np.ones([R,1])).astype('float64') 
            prior_lambda_c[k] = np.multiply( pow(10,-12), np.ones([R,1])).astype('float64') 
            Lambda[k] = np.transpose(np.matrix(np.power(rms(Y[k],0,[]),-2))).astype('float64')
            #% Note that any "missing data" scans should use Ga(b=1e-18, c=1e12)
            lambda_log[k] = np.log(Lambda[k]);
            lambda_c[k] = pow(10,12)*np.ones(1).astype('float64')
            lambda_binv[k] = np.divide(lambda_c[k],Lambda[k])
            lambda_R[k] = copy.deepcopy(Lambda[k])            
        elif opts['lambda_dims'] == 'G':
            print('not default, need to add?')            
        elif opts['lambda_dims'] == 'o': #the '' case in matlab
            prior_lambda_b[k] =  pow(10,12)*np.ones(1).astype('float64')
            prior_lambda_c[k] = pow(10,-12)*np.ones(1).astype('float64')
            Lambda[k] = np.transpose(np.matrix(np.power(rms(Y[k],[],[]),-2))).astype('float64')
            lambda_log[k] = np.log(Lambda[k]);
            lambda_c[k] = pow(10,12)*np.ones(1).astype('float64')
            lambda_binv[k] = np.divide(lambda_c[k],Lambda[k]) 
            lambda_R[k] = np.tile(Lambda[k],(R,1))
    
    # Initialize eta [LxG]:Initial updates: eta H {lambda X|q,q,X}*2
    prior_eta_b = pow(10,6)*np.ones(1).astype('float64') #1e3 * 1000;
    prior_eta_c = pow(10,-3)*np.ones(1).astype('float64')#1e-3;
    eta = np.multiply(np.multiply(prior_eta_b,prior_eta_c), np.ones([L,np.int(1)]))  
    eta_log = np.log(eta)
    eta_c = copy.deepcopy(prior_eta_c)
    eta_binv = np.divide(np.ones(1).astype('float64'),prior_eta_b) 
    
    if opts['lambda_dims'] == 'R': 
        NH = R;         
    else: 
        NH = np.int(1) #G.astype('int'); # Which is 1...need to remove
        
    
    #gather for output as dictionary
    Posteriors={"X":X,"X2":X2,"XtDX": XtDX,"Xq_var":Xq_var,                
                "W":W,"W_rowcov":W_rowcov,"WtW":WtW,
                "H":H,"H2Gmat":H2Gmat,"H_colcov": H_colcov,"HlambdaHt":HlambdaHt,
                "mu": mu,"mu2":mu2,"mu_var":mu_var,
                "beta":beta,"beta_log":beta_log,"beta_c":beta_c,"beta_binv":beta_binv,
                "pi_weights":pi_weights,"pi_mean":pi_mean,"pi_log":pi_log,
                "Lambda":Lambda,"lambda_log":lambda_log,"lambda_c":lambda_c,"lambda_binv":lambda_binv,"lambda_R":lambda_R,"lambda_log_R":lambda_log_R,
                "eta":eta,"eta_log":eta_log,"eta_c":eta_c,"eta_binv":eta_binv,
                "Gmat":Gmat,"Y2D_sumN":Y2D_sumN,"sumN_Dq":sumN_Dq,"sumN_DqXq":sumN_DqXq,"sumN_DqXq2":sumN_DqXq2,"sumN_Dqlogq":sumN_Dqlogq,
                "qq":qq}
            
    Priors={"prior_pi_weights":prior_pi_weights,"prior_beta_b":prior_beta_b,"prior_beta_c":prior_beta_c,
            "prior_mu_mean":prior_mu_mean,"prior_mu_var":prior_mu_var,
            "prior_lambda_b":prior_lambda_b,"prior_lambda_c":prior_lambda_c,
            "prior_eta_b":prior_eta_b,"prior_eta_c":prior_eta_c, "prior_W_var":prior_W_var}
    
    
    Constants={"K":K,"L":L,"R":R ,"DD":DD,"N":N,"G":G,"NH":NH}
    
    return Priors, Posteriors, Constants 


def flica_iterate(Y,opts,Priors, Posteriors, Constants):               
    #define list to keep info for free energy
    
    # Fpart = {k: zeros32(Constants['k']) for k in [listofvals]}
        
    Fpart = {"Hprior":np.zeros(1),"Hpost":np.zeros(1),
             "etaPrior":np.zeros(1),"etaPost":np.zeros(1),
             "Wprior":np.zeros(Constants['K']),"Wpost":np.zeros(Constants['K']),
             "muPrior":np.zeros(Constants['K']),"muPost":np.zeros(Constants['K']),
             "betaPrior":np.zeros(Constants['K']), "betaPost":np.zeros(Constants['K']),
             "piPrior":np.zeros(Constants['K']), "piPost":np.zeros(Constants['K']), 
             "qPrior":np.zeros(Constants['K']), "qPost":np.zeros(Constants['K']), 
             "Ylike1":np.zeros(Constants['K']), "Ylike2":np.zeros(Constants['K']), 
             "Ylike3":np.zeros(Constants['K']), "Ylike4":np.zeros(Constants['K']), 
             "lambdaPrior":np.zeros(Constants['K']),"lambdaPost":np.zeros(Constants['K']) ,
             "XPrior":np.zeros(Constants['K']),"XPost":np.zeros(Constants['K'])} 
    
    F_history = [];
    convergence_flag=0
    its=-1            
    # iterate the updates    
    while convergence_flag == 0 : 
         its=its+1
         print 'its = %s' % its
         tt=time.time()
         
 ## Update eta  
         tt2=time.time()   
 
         input_eta_update={'prior_eta_b': Priors['prior_eta_b'],'prior_eta_c': Priors['prior_eta_c'],
                           'H2Gmat':Posteriors['H2Gmat'],'Gmat':Posteriors['Gmat'],'L':Constants['L']}
         
         output_eta_dict = update_eta(input_eta_update)  
         
         Posteriors['eta_binv']=output_eta_dict['eta_binv']
         Posteriors['eta_c']=output_eta_dict['eta_c']
         Posteriors['eta']=output_eta_dict['eta']
         Posteriors['eta_log']=output_eta_dict['eta_log']
         print 'Time of eta =', time.time()-tt2

         
## Update H : depends on lamda_dims (R or ) and iterates over K
         tt2=time.time()   

         input_H_update={'opts':opts,'Y':Y,'NH':Constants['NH'],
                         'X':Posteriors['X'],'H':Posteriors['H'],'W':Posteriors['W'],
                         'eta':Posteriors['eta'],'Gmat':Posteriors['Gmat'],'Lambda':Posteriors['Lambda'],'lambda_R':Posteriors['lambda_R'],
                         'XtDX':Posteriors['XtDX'] ,'WtW':Posteriors['WtW'],
                         'K':Constants['K'],'R':Constants['R'], 'L':Constants['L'],'DD':Constants['DD']}
         
         output_H_dict = update_H(input_H_update)
         
         Posteriors['H']=output_H_dict['H']
         Posteriors['H2Gmat']=output_H_dict['H2Gmat']
         Posteriors['H_colcov']=output_H_dict['H_colcov']
         Posteriors['W']=output_H_dict['W']
         Posteriors['H_PCs']=output_H_dict['H_PCs']
         Posteriors['tmp_R_to_NH']=output_H_dict['tmp_R_to_NH']          
         print 'Time of H =', time.time()-tt2

         
#update H*lambda{k}*H'> and also W : both iterate over K together; update W requires hugh matrix mult
         tt2=time.time()
         input_HlamW_update={'Y':Y,'X':Posteriors['X'],'H':Posteriors['H'],'W':Posteriors['W'],
                            'K':Constants['K'],'R':Constants['R'], 'L':Constants['L'],'DD':Constants['DD'],
                            'HlambdaHt':Posteriors['HlambdaHt'],'lambda_R':Posteriors['lambda_R'],'H_colcov':Posteriors['H_colcov'],
                            'Gmat':Posteriors['Gmat'],'XtDX':Posteriors['XtDX'] ,'WtW':Posteriors['WtW'],
                            'prior_W_var':Priors['prior_W_var'] ,'W_rowcov':Posteriors['W_rowcov']}
         
         output_HlamW_dict = update_HlambdaHt_and_W(input_HlamW_update)
         
         Posteriors['HlambdaHt']=output_HlamW_dict['HlambdaHt']
         Posteriors['W']=output_HlamW_dict['W']
         Posteriors['WtW']=output_HlamW_dict['WtW']
         Posteriors['W_rowcov']=output_HlamW_dict['W_rowcov']
         print 'Time of HlambdaHt =', time.time()-tt2
         
         
         
         
#Update X: ITERATES OVER K AND OVER L
         
         tt2=time.time() 
         for k in range(Constants['K']):
             input_X_k_update={'X_k':Posteriors['X'][k],'H':Posteriors['H'], 'Y_k':Y[k],'L':Constants['L'],
                              'DD_k':Constants['DD'][k],'X2_k':Posteriors['X2'][k],'lambda_R_k':Posteriors['lambda_R'][k],
                              'W_k':Posteriors['W'][k],'WtW_k':Posteriors['WtW'][k],'HlambdaHt_k':Posteriors['HlambdaHt'][k],
                              'beta_k':Posteriors['beta'][k],'mu_k':Posteriors['mu'][k],
                              'Xq_var_k':Posteriors['Xq_var'][k],'sumN_Dq_k':Posteriors['sumN_Dq'][k,:,:],'sumN_DqXq_k':Posteriors['sumN_DqXq'][k,:,:],
                              'sumN_DqXq2_k':Posteriors['sumN_DqXq2'][k,:,:],'sumN_Dqlogq_k':Posteriors['sumN_Dqlogq'][k,:,:],
                              'beta_log_k':Posteriors['beta_log'][k],'mu2_k':Posteriors['mu2'][k],'pi_log_k':Posteriors['pi_log'][k]}

            ##@jit(nopython=True, parallel=True)
             output_X_k_dict= update_X_k(input_X_k_update)

             Posteriors['X'][k]=output_X_k_dict['X_k']
             Posteriors['X2'][k]=output_X_k_dict['X2_k']
             Posteriors['sumN_Dqlogq'][k]=output_X_k_dict['sumN_Dqlogq_k']
             Posteriors['sumN_DqXq2'][k]=output_X_k_dict['sumN_DqXq2_k']
             Posteriors['sumN_DqXq'][k]=output_X_k_dict['sumN_DqXq_k']
             Posteriors['sumN_Dq'][k]=output_X_k_dict['sumN_Dq_k']
             Posteriors['Xq_var'][k]=output_X_k_dict['Xq_var_k']
             
         print 'Time of X ', time.time()-tt2    

         
         

#%% UPDATE THE MIXTURE MODELS         
         tt2=time.time()   

         input_mixmod_update={'X':Posteriors['X'],'X2':Posteriors['X2'],'XtDX':Posteriors['XtDX'],
                     'K':Constants['K'], 'DD':Constants['DD'],
                     'pi_weights':Posteriors['pi_weights'],'prior_pi_weights':Priors['prior_pi_weights'],'sumN_Dq':Posteriors['sumN_Dq'], 
                     'pi_mean':Posteriors['pi_mean'],'pi_log':Posteriors['pi_log'],
                     'beta':Posteriors['beta'],'beta_log':Posteriors['beta_log'],'beta_c':Posteriors['beta_c'],'prior_beta_c':Priors['prior_beta_c'] ,
                     'beta_binv':Posteriors['beta_binv'],'prior_beta_b':Priors['prior_beta_b'], 
                     'mu':Posteriors['mu'],'mu2':Posteriors['mu2'],'sumN_DqXq':Posteriors['sumN_DqXq'],'sumN_DqXq2':Posteriors['sumN_DqXq2'],
                     'mu_var':Posteriors['mu_var'],'prior_mu_var':Priors['prior_mu_var'] ,'prior_mu_mean':Priors['prior_mu_mean'],
                         }  
         output_mixmod_dict=update_mixmod(input_mixmod_update)
         
         Posteriors['XtDX']=output_mixmod_dict['XtDX']
         Posteriors['pi_weights']=output_mixmod_dict['pi_weights']
         Posteriors['pi_log']=output_mixmod_dict['pi_log']
         Posteriors['pi_mean']=output_mixmod_dict['pi_mean']
         Posteriors['beta_c']=output_mixmod_dict['beta_c']
         Posteriors['beta_binv']=output_mixmod_dict['beta_binv']
         Posteriors['beta']=output_mixmod_dict['beta']
         Posteriors['beta_log']=output_mixmod_dict['beta_log']
         Posteriors['mu']=output_mixmod_dict['mu']
         Posteriors['mu_var']=output_mixmod_dict['mu_var']
         Posteriors['mu2']=output_mixmod_dict['mu2']
         print 'Time of Mix model ', time.time()-tt2    

         
#%% Update P'(lambda)
         tt2=time.time()   

         input_lambda_update={'opts':opts,'Y':Y,'X':Posteriors['X'],'H':Posteriors['H'],'W':Posteriors['W'],
                         'K':Constants['K'], 'L':Constants['L'],'DD':Constants['DD'],'R':Constants['R'],'N':Constants['N'],                        
                         'WtW':Posteriors['WtW'],'XtDX':Posteriors['XtDX'], 'H_colcov':Posteriors['H_colcov'],'Gmat':Posteriors['Gmat'],
                         'lambda_c':Posteriors['lambda_c'],'lambda_binv':Posteriors['lambda_binv'],
                         'Lambda':Posteriors['Lambda'],'lambda_log':Posteriors['lambda_log'],
                         'lambda_log_R':Posteriors['lambda_log_R'],'lambda_R':Posteriors['lambda_R'],'HlambdaHt':Posteriors['HlambdaHt'],
                         'prior_lambda_c':Priors['prior_lambda_c'],'prior_lambda_b':Priors['prior_lambda_b']}                         
                                                 
         #tt2=time.time()
         output_lambda_dict = update_lambda(input_lambda_update)
         
         Posteriors['Lambda']=output_lambda_dict['Lambda']
         Posteriors['HlambdaHt']=output_lambda_dict['HlambdaHt']
         Posteriors['lambda_log_R']=output_lambda_dict['lambda_log_R']
         Posteriors['lambda_R']=output_lambda_dict['lambda_R']
         Posteriors['lambda_log']=output_lambda_dict['lambda_log']
         Posteriors['lambda_binv']=output_lambda_dict['lambda_binv']
         Posteriors['lambda_c']=output_lambda_dict['lambda_c']
         print 'Time of Lambda', time.time()-tt2    
                           
#%% Compute F, if desired
         if opts['computeF']==1:
             input_FE_computation={'Fpart':Fpart, 'Y':Y,'X':Posteriors['X'],'H':Posteriors['H'],'W':Posteriors['W'],
                             'K':Constants['K'], 'L':Constants['L'],'DD':Constants['DD'],'R':Constants['R'],'N':Constants['N'],'G':Constants['G'], 
                             'H_colcov':Posteriors['H_colcov'],'H2Gmat':Posteriors['H2Gmat'],'W_rowcov':Posteriors['W_rowcov'],
                             'WtW':Posteriors['WtW'],'mu2':Posteriors['mu2'],'mu_var':Posteriors['mu_var'],'Gmat':Posteriors['Gmat'],
                             'beta':Posteriors['beta'],'beta_log':Posteriors['beta_log'],'beta_c':Posteriors['beta_c'],'beta_binv':Posteriors['beta_binv'],
                             'pi_log':Posteriors['pi_log'],'pi_weights':Posteriors['pi_weights'],                         
                             'eta':Posteriors['eta'],'eta_c':Posteriors['eta_c'],'eta_binv':Posteriors['eta_binv'],'eta_log':Posteriors['eta_log'],
                             'sumN_Dqlogq':Posteriors['sumN_Dqlogq'],'lambda_log_R':Posteriors['lambda_log_R'],
                             'Y2D_sumN':Posteriors['Y2D_sumN'],'lambda_R':Posteriors['lambda_R'],'lambda_log':Posteriors['lambda_log'],
                             'lambda_binv':Posteriors['lambda_binv'],'lambda_c':Posteriors['lambda_c'],'Lambda':Posteriors['Lambda'],
                             'HlambdaHt':Posteriors['HlambdaHt'],'sumN_Dq':Posteriors['sumN_Dq'],'XtDX':Posteriors['XtDX'],
                             'mu':Posteriors['mu'],'sumN_DqXq':Posteriors['sumN_DqXq'],'sumN_DqXq2':Posteriors['sumN_DqXq2'],'Xq_var':Posteriors['Xq_var'],
                             'prior_eta_b': Priors['prior_eta_b'],'prior_eta_c': Priors['prior_eta_c'],
                             'prior_W_var':Priors['prior_W_var'] ,'prior_mu_var':Priors['prior_mu_var'] ,
                             'prior_beta_c':Priors['prior_beta_c'],'prior_beta_b':Priors['prior_beta_b'],
                             'prior_pi_weights':Priors['prior_pi_weights'],'prior_lambda_c':Priors['prior_lambda_c'],
                             'prior_lambda_b':Priors['prior_lambda_b']}
                                                     
             #tt2=time.time()
             F, Fpart = compute_F(input_FE_computation)
             F_history.append(F);
             #print 'cost_F =', time.time()-tt2
             print 'F =', F       
         else:
             F_history.append(9999)
             F=9999
             print 'F = not computed...'
             if its > (opts['maxits']-2):
                 input_FE_computation={'Fpart':Fpart, 'Y':Y,'X':Posteriors['X'],'H':Posteriors['H'],'W':Posteriors['W'],
                     'K':Constants['K'], 'L':Constants['L'],'DD':Constants['DD'],'R':Constants['R'],'N':Constants['N'],'G':Constants['G'], 
                     'H_colcov':Posteriors['H_colcov'],'H2Gmat':Posteriors['H2Gmat'],'W_rowcov':Posteriors['W_rowcov'],
                     'WtW':Posteriors['WtW'],'mu2':Posteriors['mu2'],'mu_var':Posteriors['mu_var'],'Gmat':Posteriors['Gmat'],
                     'beta':Posteriors['beta'],'beta_log':Posteriors['beta_log'],'beta_c':Posteriors['beta_c'],'beta_binv':Posteriors['beta_binv'],
                     'pi_log':Posteriors['pi_log'],'pi_weights':Posteriors['pi_weights'],                         
                     'eta':Posteriors['eta'],'eta_c':Posteriors['eta_c'],'eta_binv':Posteriors['eta_binv'],'eta_log':Posteriors['eta_log'],
                     'sumN_Dqlogq':Posteriors['sumN_Dqlogq'],'lambda_log_R':Posteriors['lambda_log_R'],
                     'Y2D_sumN':Posteriors['Y2D_sumN'],'lambda_R':Posteriors['lambda_R'],'lambda_log':Posteriors['lambda_log'],
                     'lambda_binv':Posteriors['lambda_binv'],'lambda_c':Posteriors['lambda_c'],'Lambda':Posteriors['Lambda'],
                     'HlambdaHt':Posteriors['HlambdaHt'],'sumN_Dq':Posteriors['sumN_Dq'],'XtDX':Posteriors['XtDX'],
                     'mu':Posteriors['mu'],'sumN_DqXq':Posteriors['sumN_DqXq'],'sumN_DqXq2':Posteriors['sumN_DqXq2'],'Xq_var':Posteriors['Xq_var'],
                     'prior_eta_b': Priors['prior_eta_b'],'prior_eta_c': Priors['prior_eta_c'],
                     'prior_W_var':Priors['prior_W_var'] ,'prior_mu_var':Priors['prior_mu_var'] ,
                     'prior_beta_c':Priors['prior_beta_c'],'prior_beta_b':Priors['prior_beta_b'],
                     'prior_pi_weights':Priors['prior_pi_weights'],'prior_lambda_c':Priors['prior_lambda_c'],
                     'prior_lambda_b':Priors['prior_lambda_b']}
                 F, Fpart = compute_F(input_FE_computation)
                 F_history.append(F);
                 print 'F final = ', F
    
         if its>0:
            dF = (F - F_history[its-1])#/(its-tmpPrevIt);
            print 'Difference of F between iteration = ',float(dF)/float(np.abs(F))
            
            #import pdb;pdb.set_trace()
            if its > (opts['maxits']-1): #| (dF<
                convergence_flag=1
                #GATHER OUTPUT
         FLICA_OUTPUT_DICT ={"H":Posteriors['H'],"lambda":Posteriors['Lambda'],"W":Posteriors['W'],"beta":Posteriors['beta'],"mu":Posteriors['mu'],
                "pi":Posteriors['pi_mean'],"X":Posteriors['X'], "H_PCs":Posteriors['H_PCs'],"F":F,"F_history":F_history,
                               "DD":Constants['DD'],"opts":opts}
              
         print 'cost_per_iteration =', time.time()-tt, ' seconds...'
         
         output_dir=opts['output_dir']
         iters_to_save=np.array(range(25,opts['maxits']+1,25));
         if sum(its==iters_to_save)==1:
             if os.path.exists(os.path.join(output_dir,'iter'+str(its)) )==0:
                 os.mkdir( os.path.join(output_dir,'iter'+str(its))  )
             np.savez(os.path.join(output_dir,'iter'+str(its),'flica_result.npz'),DD=FLICA_OUTPUT_DICT['DD'],F=FLICA_OUTPUT_DICT['F'],
                      F_history=FLICA_OUTPUT_DICT['F_history'],
                      H=FLICA_OUTPUT_DICT['H'],H_PCs=FLICA_OUTPUT_DICT['H_PCs'],
                      W=FLICA_OUTPUT_DICT['W'],beta=FLICA_OUTPUT_DICT['beta']
                      ,lambda1=FLICA_OUTPUT_DICT['lambda'],mu=FLICA_OUTPUT_DICT['mu'],pi=FLICA_OUTPUT_DICT['pi'],
                      NH=Constants['NH'],eta=Posteriors['eta'],Gmat=Posteriors['Gmat'],lambda_R=Posteriors['lambda_R'],
                      XtDX=Posteriors['XtDX'],WtW=Posteriors['WtW'],K=Constants['K'],R=Constants['R'],L=Constants['L'])
             
             for ii in range(0,len(Posteriors['X'])):
                 np.save(os.path.join(output_dir,'iter'+str(its),'flica_X'+str(ii+1)+'.npy'),Posteriors['X'][ii])
             
             os.system('rm -r '+os.path.join(output_dir,'iter'+str(its-25)))
            
            
    return FLICA_OUTPUT_DICT # the old M 


