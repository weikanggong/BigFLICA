# BigFLICA v0.1

## Python code for BigFLICA
Gong, Weikang, Christian F. Beckmann, and Stephen M. Smith. "Phenotype Discovery from Population Brain Imaging." bioRxiv (2020).

## Requirements
Python 2.7.x, spams (for dictionary learning), numpy, scipy, pylab, copy.
System tested: Linux 

## Usage
The main function to use is the BigFLICA function in the script BigFLICA_cpu.py. Put this module in a position where your python can find it (e.g., /home/weikanggong). Then, prepare the data as .npy files (assume that we have two modalities, and they are stored as /home/weikanggong/mod1.npy, /home/weikanggong/mod2.npy). Finally, suppose the output directory is /home/weikanggong/bigflica_output. 

Example code is something like the following:
```
import BigFLICA_cpu
data_loc = ['/home/weikanggong/mod1.npy',
            '/home/weikanggong/mod2.npy']
output_dir = '/home/weikanggong/bigflica_output/'
nlat = 10
migp_dim =100
dicl_dim =500
ncore = 1

BigFLICA_cpu.BigFLICA(data_loc, nlat, output_dir, migp_dim, dicl_dim, ncore)

```
1. data_loc: a list whose length equals to the number of modalities, each element is the absolute directory of data
          matrix of one modality in .npy format. The data matrix is assumed to be of size subject * voxels (This can be generated by vectorizing the voxel dimension by appling a binary mask).
          The number of subjects should be equal across modalities. (Subjects with a missing modality can be imputed by the             mean of other subjects).
2. nlat: Number of components to extract in BigFLICA
3. output_dir: the absolute directory to store all BigFLICA results
4. migp_dim: Number of components to extract in MIGP step (migp_dim > nlat).
5. dicl_dim: Number of components to extract in the Dictionary learning step
6. ncore: Number of CPUs to perform dictionary learning on each modality.

## Outputs
In the specified output directory,
1. subj_course.npy is the subject course (H matrix in the paper), which is of the size subject-by-FLICA_component, this is the matrix used to correlate with behavioural variables, or to predict the behavioural variables.
2. flica_mod*_Z.npy is the Z-score normalized spatial maps of each modality, which is of the size voxel-by-FLICA_component.
3. mod_contribution.npy is the relative contribution of each modality to each FLICA component, which is of the size modality-by-FLICA_component. Within each FLICA component, the contribution of different modalities can be sorted based on these numbers.


## Things will be added in the future release:
1. Functions support nifti, cifti and freesurfer inputs.
2. Plotting the spatial maps.







