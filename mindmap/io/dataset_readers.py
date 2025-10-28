"""
Module Name: dataset_readers.py
Description: Classes to read data from specific dataset and `dataset_reader()` function
to point towards correct class given a configuration file.

This module provides dataset readers for specific datasets. Every reader is build upon the 
DatasetReader class which manages dataset meta-information. Every class indicates how to access
a specific data by id. 

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import numpy as np
import scipy as sp
import nibabel as nib
import mne

from mindmap.io.readers import DatasetReader, read_config, read_eeg, read_mri

def dataset_reader(config):
    """
    Returns class that reads specific dataset.

    Parameter
    ---------
    config: str | dict
        Configuration. Includes dataset name.

    Returns
    -------
    reader: DatasetReader
        Specific dataset reader.
    """
    if isinstance(config, str):
        config = read_config(config)

    dataset = config["NAME"]

    if dataset.lower() == "spike":
        return SpikeReader(config)
    elif dataset.lower() == "simf":
        return SimfReader(config)
    else:
        raise ValueError(f"Unknown dataset '{dataset}' for dataset reader.")



# --------------
# SPIKE dataset
# --------------


class SpikeReader(DatasetReader):
    """
    Class to read data from the SPIKE dataset.
    """
    def __init__(self, config):
        super(SpikeReader, self).__init__(config)

    def __call__(self, subject, *args, mri_dtype="nib", sample=None, verbose=1):
        """
        Read data for dataset SPIKE.

        Parameters
        ----------
        subject: str | int
            Subject name or id.

        *args: list(str)
            Requested data. Data are requested via an abreviated code. See Returns to check 
            available abreviations and associated datatype.

        mri_dtype: str
            MRI dtype (Nibabel or ANTs) for MRI images.

        Returns:
        data: dict
            Dictionnary containing (only) requested data:
            - t1: nib.Nifti1Image | ants.ANTsImage
                Presurgery T1-weigthed MRI.
            - t1-post: nib.Nifti1Image | ants.ANTsImage
                Postsurgery T1-weighted MRI.
            - resec: nib.Nifti1Image | ants.ANTsImage
                Resection area.
            - eeg: mne.Evoked
                Averaged inter-ictal EEG.
            - eeg-raw:mne.Raw
                Raw inter-ictal EEG.
            - t1-fs: nib.Nifti1Image | ants.ANTsImage
                T1-weigthed MRI in freesurfer space (tkras).
            - t1-brain: nib.Nifti1Image | ants.ANTsImage
                Brain mask.
            - white: list[np.ndarray (n,3), np.ndarray (m,3)]
                White surface (boundary between white a gray matter). The surface correponds
                to a mesh defined by its vertex locations (n,3) and triangular faces (m,3).
            - charm: nib.Nifti1Image | ants.ANTsImage
                CHARM head segmentation.
            - tkras2ras: np.ndarray (4,4)
                Affine transformation (freesurfer space (tkras) -> T1 ras space)
            - parc: Parcellation
                Parcellation (T1-ras space)
            - ctx: nib.Nifti1Image
                Cortical mask.
            - src: mne.SourceSpaces
                Source space.
            - src-raw: mne.SourceSpaces
                Source space before forward solution.
            - bem: mne.BEM
                Boundary element model.
            - bem-sim: mne.BEM
                Boundary element model for simulations.
            - fwd: mne.Forward
                Forward solution
            - sc-nfiber: np.ndarray
                Structural connectome with number of fibers as weights.
            - sc-length: np.ndarray
                Structural connectome with fiber lengths as weights.
            - sc-rb-nfiber: np.ndarray
                Region-based structural connectome with number of fibers as weights.
            - sc-rb-nfiber: np.ndarray
                Region-based structural connectome with fiber lengths as weights.
            - eeg: mne.Evoked
                Averaged and preprocessed EEG.
            - cset: mne.SourceEstimate
                CSET inverse solution.
            - sloreta: mne.SourceEstimate
                sLORETA inverse solution.
            - wcset: mne.SourceEstimate
                weighted CSET inverse solution.
            - trans: mne.Transformation
                Affine transformation (sensor space -> freesurfer space)
        """
        if isinstance(subject, int):
            subject = self.config["SUBS"][subject-1]

        name   = self.config["NAME"]
        ds_dir = self.config["DS_DIR"]
        mm_dir = self.config["MM_DIR"]

        data = {
            "name":name,
            "sub":subject,
            "samp":"spike"
        }

        # Others
        for key in args:

            # -------------
            # Read Anatomy
            # -------------

            # Presurgery T1-weighted MRI
            if key.lower() == "t1":
                file = ds_dir/"derivatives"/"cmp-v3.0.0-beta-RC1"/subject/"ses-preop"/"anat"/f"{subject}_ses-preop_space-individual_desc-brain_T1w.nii.gz"

            # Postsurgery T1-weighted MRI
            if key.lower() == "mri_post":
                file = ds_dir/"derivatives"/"cmp-v3.0.0-beta-RC1"/subject/"ses-postop"/"anat"/f"{subject}_ses-postop_space-individual_desc-brain_T1w.nii.gz"
                data[key] = read_mri(file, mri_dtype=mri_dtype)

            # Resection area
            if key.lower() == "resec":
                # mask of resected area
                manual = mm_dir/subject/"resec"/f"{subject}_man-recon-seg.nii.gz"
                auto   = mm_dir/subject/"resec"/f"{subject}_auto-seg.nii.gz"

                if manual.exists():
                    data[key] = read_mri(manual, mri_dtype=mri_dtype)
                elif auto.exists():
                    data[key] = read_mri(auto, mri_dtype=mri_dtype)
                else:
                    continue

            # ---------
            # Read EEG
            # ---------

            if key.lower() == "eeg-raw":
                folder = ds_dir/"derivatives"/"cartool-v3.80"/subject/"ses-preop"/"eeg"
                data[key] = read_eeg(folder, dataset=name)

        args = list(set(args) - set(data.keys()))    # update missing arguments

        # --------------------------------------------------
        # Read FreeSurfer derivatives
        # [t1-fs, t1-brain, white, charm, tkras2ras]
        # --------------------------------------------------

        fs_data = self.read_freesurfer(subject, *args, mri_dtype=mri_dtype) 
        data.update(fs_data)
        args = list(set(args) - set(data.keys()))

        # ------------------------------------------------------------------------
        # Read MindMap derivatives
        # [parc, ctx, , src, src-raw, bem, bem-sim, fwd, sc-nfiber, sc-length, \\
        # sc-rb-nfiber, sc-rb-length, eeg, cset, sloreta, wcset, trans]
        # ------------------------------------------------------------------------

        mm_data = self.read_mindmap(subject, *args, sample="spike")
        data.update(mm_data)
        args = list(set(args) - set(data.keys()))

        # ------------------------------------------------------------------------
        # Read Standardized data
        # [std-ctx, std-thal, std-mni]
        # ------------------------------------------------------------------------
        
        std_data = self.read_standards(*args, mri_dtype=mri_dtype)
        data.update(std_data)
        args = list(set(args) - set(data.keys()))

        if verbose == 1:
            self.log(data, args)

        return data



# --------------
# SIMF dataset
# --------------


class SimfReader(DatasetReader):
    """
    Class to read data from the SIMF dataset.
    """
    def __init__(self, config):
        super(SimfReader, self).__init__(config)

    def __call__(self, subject, *args, mri_dtype="nib", sample=None, verbose=1):
        """
        Read data for dataset SIMF.

        Parameters
        ----------
        subject: str | int
            Subject name or id.

        *args: list(str)
            Requested data. Data are requested via an abreviated code. See Returns to check 
            available abreviations and associated datatype.

        mri_dtype: str
            MRI dtype (Nibabel or ANTs) for MRI images.

        Returns:
        data: dict
            Dictionnary containing (only) requested data:
            - t1: nib.Nifti1Image | ants.ANTsImage
                Presurgery T1-weigthed MRI.
            - t1-post: nib.Nifti1Image | ants.ANTsImage
                Postsurgery T1-weighted MRI.
            - resec: nib.Nifti1Image | ants.ANTsImage
                Resection area.
            - eeg: mne.Evoked
                Averaged inter-ictal EEG.
            - eeg-raw:mne.Raw
                Raw inter-ictal EEG.
            - t1-fs: nib.Nifti1Image | ants.ANTsImage
                T1-weigthed MRI in freesurfer space (tkras).
            - t1-brain: nib.Nifti1Image | ants.ANTsImage
                Brain mask.
            - white: list[np.ndarray (n,3), np.ndarray (m,3)]
                White surface (boundary between white a gray matter). The surface correponds
                to a mesh defined by its vertex locations (n,3) and triangular faces (m,3).
            - charm: nib.Nifti1Image | ants.ANTsImage
                CHARM head segmentation.
            - tkras2ras: np.ndarray (4,4)
                Affine transformation (freesurfer space (tkras) -> T1 ras space)
            - parc: Parcellation
                Parcellation (T1-ras space)
            - ctx: nib.Nifti1Image
                Cortical mask.
            - src: mne.SourceSpaces
                Source space.
            - src-raw: mne.SourceSpaces
                Source space before forward solution.
            - bem: mne.BEM
                Boundary element model.
            - bem-sim: mne.BEM
                Boundary element model for simulations.
            - fwd: mne.Forward
                Forward solution
            - sc-nfiber: np.ndarray
                Structural connectome with number of fibers as weights.
            - sc-length: np.ndarray
                Structural connectome with fiber lengths as weights.
            - sc-rb-nfiber: np.ndarray
                Region-based structural connectome with number of fibers as weights.
            - sc-rb-nfiber: np.ndarray
                Region-based structural connectome with fiber lengths as weights.
            - eeg: mne.Evoked
                Averaged and preprocessed EEG.
            - cset: mne.SourceEstimate
                CSET inverse solution.
            - sloreta: mne.SourceEstimate
                sLORETA inverse solution.
            - wcset: mne.SourceEstimate
                weighted CSET inverse solution.
            - trans: mne.Transformation
                Affine transformation (sensor space -> freesurfer space)
        """
        if isinstance(subject, int):
            subject = self.config["SUBS"][subject-1]
        if isinstance(sample, int):
            sample = self.config["SAMP"][sample-1]

        name    = self.config["NAME"]
        mm_dir  = self.config["MM_DIR"]
        cmp_dir = self.config["CMP_DIR"]
        sim_dir = self.config["SIM_DIR"]

        data = {
            "name":name,
            "sub":subject
        }

        if sample is not None: 
            data["samp"] = sample

        if isinstance(sample, int):
            sample = f"sample{sample:03}"

        # dataset specifics (use this section to also overwrite freesurfer and mindmap)
        for key in args:

            # -------------
            # Read Anatomy
            # -------------
            
            # Tractogram
            if key.lower() == "tracto":
                if subject == "vepcon_sub-10":
                    file = cmp_dir/subject/"dwi"/"sub-10_model-CSD_desc-DET_tractogram.trk"
                else:
                    file = cmp_dir/subject/"fibtracking"/"sd_stream"/"dwi_denoised_eddy_distc_corrected_1mm_shore_sdstream.trk"
                data[key] = nib.streamlines.load(file)

            # ---------
            # Read EEG
            # ---------

            # EEG averaged
            if key.lower() == "eeg":
                if sample is None:
                    # look for empty coregistered
                    d = self.__call__(subject, "eeg-coreg", verbose=0)
                    data[key] = d["eeg-coreg"]
                else:
                    file = sim_dir/subject/sample/"eeg_ave.fif"
                    data[key] = mne.Evoked(file, verbose="warning")

            # default null EEG
            if key.lower() == "eeg-raw":
                # load builtin montage
                montage = mne.channels.make_standard_montage("EGI_256")
            
                # create info
                info    = mne.create_info(ch_names=montage.ch_names, ch_types="eeg", sfreq=250)
                info.set_montage(montage)

                info["description"] = "Simulation EEG"
                info["bads"] = ["E31", "E67", "E73", "E82", "E91", "E92", "E102", "E111", "E120", "E133", "E145", "E165", "E174", "E187", "E199", "E208", "E209", "E216", "E217", "E218", "E219"] + [f"E{i}" for i in range(225,257)]

                # create null data
                eeg = np.zeros((info["nchan"], int(info["sfreq"])))

                # create eeg
                eeg = mne.EvokedArray(eeg, info, verbose="warning")
                eeg.drop_channels(info["bads"])

                data[key] = eeg

            # coregistered null EEG
            if key.lower() == "eeg-coreg":
                file      = mm_dir/subject/"eeg"/f"{subject}_desc-empty-coreg_ave.fif"
                data[key] = mne.Evoked(file, verbose="warning")

            # ---------------------
            # Read transformations
            # ---------------------

            # EEG coregistration (sensor space in [m] -> freesurfer ras space in [m])
            if key.lower() == "trans":
                data[key] = mne.transforms.Transform("head", "mri")    # identity transformation

            # diffusion ras space in [mm] -> T1-ras space in [mm]
            if key.lower() == "t12diff":
                file      = cmp_dir/subject/"xfm"/"final0GenericAffine.mat"
                affine    = sp.io.loadmat(str(file))["AffineTransform_double_3_3"].reshape(3, 4, order="F")
                data[key] = np.concatenate((affine, np.array([[0, 0, 0, 1]])))


            # -------------------------
            # Read MindMap derivatives
            # -------------------------

            if key.lower() == "gt":
                # ground truth 
                if sample is None: 
                    raise ValueError("Require sample name to read data 'gt'.")
                file = sim_dir/subject/sample/"ground-truth-noisefree"
                data[key] = mne.read_source_estimate(file)
            
            if key.lower() == "gt_noise":
                # ground truth with noise
                if sample is None:
                    raise ValueError("Require sample name to read data 'gt'.")
                file = sim_dir/subject/sample/"ground-truth-noise"
                data[key] = mne.read_source_estimate(file)

            if key.lower() == "sc_tmp":
                file      = mm_dir/subject/"sc"/f"{subject}_sc-template.npy"
                data[key] = np.load(file)
            
            if key.lower() == "cst_tmp":
                file      = mm_dir/subject/"sc"/f"{subject}_cst-template.npy"
                data[key] = np.load(file, allow_pickle=True)

            if key.lower() == "sc-length":
                file      = mm_dir/subject/"sc"/f"{data['sub']}_sc-length-tracto.npy"
                data[key] = np.load(file, allow_pickle=True)

            if key.lower() == "sc-weight":
                file      = mm_dir/subject/"sc"/f"{data['sub']}_sc-weight-tracto.npy"
                data[key] = np.load(file, allow_pickle=True)


        args = list(set(args) - set(data.keys()))   # update list of missing arguments

        # --------------------------------------------------
        # Read FreeSurfer derivatives
        # [t1-fs, t1-brain, white, charm, tkras2ras]
        # --------------------------------------------------

        fs_data = self.read_freesurfer(subject, *args, mri_dtype=mri_dtype)
        data.update(fs_data)
        args = list(set(args) - set(data.keys()))

        # ------------------------------------------------------------------------
        # Read MindMap derivatives
        # [parc, ctx, , src, src-raw, bem, bem-sim, fwd, sc-nfiber, sc-length, \\
        # sc-rb-nfiber, sc-rb-length, eeg, cset, sloreta, wcset, trans]
        # ------------------------------------------------------------------------

        mm_data = self.read_mindmap(subject, *args, sample=sample)
        data.update(mm_data)
        args = list(set(args) - set(data.keys()))

        # ------------------------------------------------------------------------
        # Read Standardized data
        # [std-ctx, std-thal, std-mni]
        # ------------------------------------------------------------------------
        std_data = self.read_standards(*args, mri_dtype=mri_dtype)
        data.update(std_data)
        args = list(set(args) - set(data.keys()))

        if verbose == 1:
            self.log(data, args)

        return data
