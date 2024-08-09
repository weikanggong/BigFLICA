import os 
import numpy as np
import BigFLICA.FLICA_cpu as flica
import BigFLICA.utils as utils
import spams 
import scipy
from joblib import Parallel, delayed
import multiprocessing

def rms(IN, dim, options):
    if dim==[]:  #I use only this case USED FLICA LOAD!!
        out = np.sqrt(np.sum(np.square(IN))/IN.size)# dumm = alb_various.rms(Y[k],0,[])
    else:
        out = np.sqrt(np.divide(np.sum(np.square(IN),dim),IN.shape[dim]))# dumm = alb_various.rms(Y[k],0,[])
    return out

def BigFLICA(data_loc, nlat, output_dir, migp_dim, dicl_dim, ncore = 1):
    '''
    data_loc: a list with length equals to the number of modalities, each is the absolute directory of data
              matrix of one modality in .npy format. The data matrix is assumed to be subject * voxels.
              The number of subjects should be equal across modalities.
    nlat: Number of components to extract in BigFLICA
    output_dir: the absolute directory to store all BigFLICA results
    migp_dim: Number of components to extract in MIGP step (migp_dim > nlat).
    dicl_dim: Number of components to extract in the Dictionary learning step
    '''
    
    os.system('mkdir '+output_dir)
    os.system('mkdir '+output_dir + '/MIGP' + str(migp_dim))
    os.system('mkdir '+output_dir + '/DicL' + str(dicl_dim))
    os.system('mkdir '+output_dir + '/Result_IC' + str(nlat))
    
    tmp = np.load(data_loc[0])#[select_index,:]
    nmod = len(data_loc)
    nsubj = tmp.shape[0]    
    
    print('MIGP...')
    if os.path.exists(output_dir+'/MIGP' + str(migp_dim) + '/U_mMIGP.npy') == 0:
        cov_mat = np.zeros((nsubj,nsubj))
        for i in range(0,nmod):
                
            print(('Loading data of Modal '+str(i+1)))        
            Data = np.load(data_loc[i])#[select_index,:]
            ind=np.sum(Data,axis=1)!=0        
            Data[ind,:]=utils.nets_zscore(Data[ind,:])
            #print('zscore normalization...')    
            #Data = utils.nets_zscore(Data)    
            print('Computing covariance matrix...')
            cov_mat = cov_mat + np.dot(Data,Data.T)/Data.shape[1]
        print('Done...')
        
        #if nsubj<50000:
        dd,uu=scipy.linalg.eigh(cov_mat,eigvals=(nsubj-migp_dim,nsubj-1))
        dd=np.real(dd)
        uu=np.real(uu)
        indx1=np.argsort(-dd)
        dd=dd[indx1]
        uu=uu[:,indx1]
        U = np.dot(uu,np.diag(dd))
        #else:
        #    U = utils.SingleModality_MIGP(cov_mat, migp_dim  ,200 , 5 ) #subj-by-migp_dim
                
        np.save(output_dir+'/MIGP' + str(migp_dim) + '/U_mMIGP.npy',U)
    else:
        U = np.load(output_dir+'/MIGP' + str(migp_dim) + '/U_mMIGP.npy')  
        
    if os.path.exists(output_dir+'/MIGP' + str(migp_dim) + '/PCAdata_mod'+('%02d' % (nmod))+'.npy') == 0:   
        for i in range(0,nmod):        
            print(('Loading data of Modal '+str(i+1)))        
            Data=np.load(data_loc[i])#[select_index,:] 
            ind=np.sum(Data,axis=1)!=0        
            Data[ind,:]=utils.nets_zscore(Data[ind,:])
            #print('zscore normalization...')    
            #Data = utils.nets_zscore(Data)          
            data_to_use=np.dot(Data.T,U) #voxel * migp_dim        
            np.save(output_dir+'/MIGP' + str(migp_dim) + '/PCAdata_mod'+('%02d' % (i+1))+'.npy',data_to_use)
    
    
    print('MIGP done...')
    
    #DicL parameters
    print('DicL...')
    
    
    param = { 'K' : dicl_dim, # learns a dictionary with dicl_dim elements, you can change this
              'lambda1' : 1,'lambda2': 1 , 'numThreads' : 1, 'batchsize' : 512,
              'iter' : 50}
    
    #one can parallize this step        
    def DicL(i,output_dir,dicl_dim,migp_dim):
        #load migp data, a subject-by-feature matrix
        ff = output_dir+'/DicL' + str(dicl_dim) + '/DLdata_mod'+('%02d' % (i+1))+'.npy'
        if os.path.exists(ff) == 0:
            dd=utils.nets_zscore(np.load(output_dir+'/MIGP' + str(migp_dim) + '/PCAdata_mod'+('%02d' % (i+1))+'.npy')) 
            D = spams.trainDL(np.asfortranarray(dd.T,dtype ='float64'),**param) #this is dicl_dim-by-migp_dim    
            np.save(ff,D)
        else:
            D = np.load(ff)
        return D   
        
    Data = Parallel(n_jobs=ncore)(delayed(DicL)(i,output_dir,dicl_dim,migp_dim) for i in range(0,nmod))
    print((len(Data)))
    print((Data[0].shape))
    
    print('DicL done ...')
    
    print('FLICA...')
    
    for i in range(0,nmod):
        Data[i]=(Data[i] - Data[i].mean(axis=0))
        Data[i]=np.divide(Data[i],rms(Data[i],[],[])).T
        
    print('Priors ...')
    opts = {"num_components":nlat,"maxits":1000,'lambda_dims':'o'
            ,"initH":"PCA","dof_per_voxel":'auto_eigenspectrum','computeF':0,
            'output_dir':output_dir+ '/Result_IC' + str(nlat)} 
    
    Priors, Posteriors, Constants=flica.flica_init_params(Data,opts)
        
    print('Performing Inference ...')
    Morig = flica.flica_iterate(Data,opts,Priors,Posteriors,Constants)
    
    print('reorder FLICA components...')
    new_order= utils.flica_reorder(output_dir+ '/Result_IC' + str(nlat),nmod)
    
    M=np.load(output_dir + '/Result_IC' + str(nlat) + '/flica_result.npz')  
    H = M['H'][new_order,:].T
    
    
    subj_course = np.dot(U,H)
    np.save(output_dir + '/Result_IC' + str(nlat) + '/subj_course.npy', subj_course)
    contri = M['H_PCs'][:,new_order]
    np.save(output_dir + '/Result_IC' + str(nlat) + '/mod_contribution.npy', contri[0:nmod,:])

    print('Z-stat spatial maps...')
    for i in range(0,nmod):
        Data=np.load(output_dir+'/MIGP'+ str(migp_dim) + '/PCAdata_mod'+('%02d' % (i+1))+'.npy')
        z_map = utils.sKPCR_regression(H,Data.T,np.ones((migp_dim,1)))
        np.save(output_dir+ '/Result_IC' + str(nlat) + '/flica_mod'+str(i+1)+'_Z.npy',z_map)
    
    return 
