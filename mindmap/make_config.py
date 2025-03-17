# Script to generate a yml configuration file

# Indications:
# - Fill the variable section and run the script
# - All variables should be in capital letters except for the `config_name`.
# - Comment lines containing variables you don't need

##########################
### Start of Variables ###
##########################
#%%
# name of configuration file
config_name = "../configurations/simf.yml"

# Meta Data
NAME = "simf"                                                                                                                           # dataset name
SUBS = ["vepcon_sub-10"]                                                                                                                # list of subject ids
SAMP = [f"sample{i+1}" for i in range(25)]                                                                                         # list of samples

# Directories
DS_DIR  = '/media/nino/DATA/SIMF'                                                                                                       # dataset
FS_HOME = '/home/nino/Softwares/freesurfer'                                                                                             # freesurfer home
FS_DIR  = '/media/nino/DATA/SIMF/derivatives/freesurfer'                                                                                # freesurfer directory
MM_DIR  = '/media/nino/DATA/SIMF/derivatives/mindmap'                                                                                   # mindmap directory
CMP_DIR = '/media/nino/DATA/SIMF/derivatives/cmp-v3.0.3'                                                                                # Connectome mapper
SIM_DIR = '/media/nino/DATA/SIMF/derivatives/simulations'                                                                               # Simulations
MONTAGE = '/media/nino/DATA/SPIKE/derivatives/cartool-v3.80/sub-pat44/ses-preop/eeg/'                                                   # Montage

# Files
CONN  = "/media/nino/DATA/atlas/probconnatlas/wm.connatlas.scale4.h5"                                                                   # connectome atlas
LAUSL = "/media/nino/DATA/atlas/laus2018/lh.atlas-laus2018_desc-scale4.annot"                                                           # Lausanne 2018 annot parcellation (left)
LAUSR = "/media/nino/DATA/atlas/laus2018/rh.atlas-laus2018_desc-scale4.annot"                                                           # Lausanne 2018 annot parcellation (right)
THAL  = "/media/nino/DATA/atlas/Thalamic_Nuclei/Thalamus_Nuclei-HCP-4DSPAMs.nii.gz"                                                     # Thalamus parcellation
THAL_CLUT = "/media/nino/DATA/atlas/Thalamic_Nuclei/Thalamic_Nuclei-ColorLUT.txt"                                                       # Thalamus CLUT 
MNI   = "/media/nino/DATA/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii"     # MNI template



########################
### END of Variables ###
########################

#%%
import yaml

data = dict()

for var in dir():
    if var.isupper():
        data[var] = eval(var)

with open(config_name, "w") as file:
    yaml.dump(data, file)
# %%
