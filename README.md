# BigFLICA v0.1

## Python code for BigFLICA
Gong, Weikang, Christian F. Beckmann, and Stephen M. Smith. "Phenotype Discovery from Population Brain Imaging." bioRxiv (2020).

## Requirements
Python 2.7.x, spams (for dictionary learning), numpy, scipy, pylab, copy.

## Usage

1. Put this module in a position where your python can find. Then, some example code are
```
from BigFLICA import BigFLICA_cpu
data_loc = ['/*/modality1.npy',
           '/*/modality2.npy']
output_dir = '/path/for/output'
nlat = 10
migp_dim =100
dicl_dim =500

BigFLICA_cpu.BigFLICA(data_loc, nlat, output_dir, migp_dim, dicl_dim)

```
1. data_loc: a list with length equals to the number of modalities, each is the absolute directory of data
          matrix of one modality in .npy format. The data matrix is assumed to be subject * voxels.
          The number of subjects should be equal across modalities.
2. nlat: Number of components to extract in BigFLICA
3. output_dir: the absolute directory to store all BigFLICA results
4. migp_dim: Number of components to extract in MIGP step (migp_dim > nlat).
5. dicl_dim: Number of components to extract in the Dictionary learning step


## Outputs
In the specified output directory,
1. subj_course.npy is the H matrix in the paper, which is of the size subject-by-FLICA_component, this is the matrix used to correlate with behaviour variables.
2. flica_mod*_Z.npy is the Z-score normalized spatial maps of one modality, which is of the size voxel-by-FLICA_component.
3. mod_contribution.npy is the contribution of each modality to each FLICA component, which is of the size modality-by-FLICA_component. Within each FLICA component, the contribution of different modalities can be sorted based on these numbers.


## Things will be added in the future release:
1. Functions support nifti, cifti and freesurfer inputs.
2. Plotting the spatial maps.







