from .readers import DatasetReader, read_config, read_eeg, read_mri

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



##########################################################
####################                  ####################
####################  SPIKE DATASET   ####################
####################                  ####################
##########################################################


class SpikeReader(DatasetReader):
    def __init__(self, config):
        super(SpikeReader, self).__init__(config)

    def __call__(self, subject, *args, mri_dtype="nib", sample=None, verbose=1):
        """
        Read data for dataset SPIKE.

        Parameters
        ----------
        subject: str
            Subject name

        *args: list(str)
            Data ids:
            - eeg: EEG (looks for eeg_raw if no EEG in mindmap folder).
            - eeg_raw: EEG from the original dataset.
            - resec: Resection area
            - mri_post: Postsurgery T1w MRI.
            For the rest, check read_freesurfer and read_mindmap

        mri_dtype: str
            MRI dtype (Nibabel or ANTs) for MRI images.

        Returns:
        data: dict
            Dictionnary of data using data ids as keys.
        """
        if isinstance(subject, int):
            subject = self.config["SUBS"][subject-1]
        
        name   = self.config["NAME"]
        ds_dir = self.config["DS_DIR"]
        mm_dir = self.config["MM_DIR"]

        data = dict(name=name, sub=subject)

        # Others
        for key in args:
                
            if key.lower() == "eeg":
                file = mm_dir/subject/"eeg"/f"{subject}-spike_ave.fif"
                if file.exists():
                    continue    # will read with read_mindmap() 
                else:
                    print(f"\033[93mPreprocessed EEG not found, reading raw instead.\033[93m")
                    d = self.__call__(subject, "eeg-raw")
                    data[key] = d["eeg-raw"]

            if key.lower() == "eeg-raw":
                folder = ds_dir/"derivatives"/"cartool-v3.80"/subject/"ses-preop"/"eeg"
                data[key] = read_eeg(folder, dataset=name)
            
            if key.lower() == "resec":
                # mask of resected area
                manual = mm_dir/subject/"resec"/f"{subject}_man-recon-seg.nii.gz"
                auto   = mm_dir/subject/"resec"/f"{subject}_auto-seg.nii.gz"

                if manual.exists() == True:
                    data[key] = read_mri(manual, mri_dtype=mri_dtype)
                elif auto.exists() == True:
                    data[key] = read_mri(auto, mri_dtype=mri_dtype)
                else:
                    continue

            if key.lower() == "mri_post":
                file = ds_dir/"derivatives"/"cmp-v3.0.0-beta-RC1"/subject/"ses-postop"/"anat"/f"{subject}_ses-postop_space-individual_desc-brain_T1w.nii.gz"
                data[key] = read_mri(file, mri_dtype=mri_dtype)

        args = list(set(args) - set(data.keys()))    # update missing arguments

        # FreeSurfer
        d    = self.read_freesurfer(subject, *args, mri_dtype=mri_dtype) 
        data.update(d)                                                 
        args = list(set(args) - set(data.keys()))                       

        # Mindmap
        d   = self.read_mindmap(subject, *args, sample="spike")
        data.update(d)
        args = list(set(args) - set(data.keys()))

        if verbose == 1:
            self.log(data, args)

        return data



##########################################################
####################                  ####################
####################  SIMF DATASET    ####################
####################                  ####################
##########################################################


class SimfReader(DatasetReader):
    def __init__(self, config):
        super(SimfReader, self).__init__(config)

    def __call__(self, subject, *args, mri_dtype="nib", sample=None, verbose=1):
        """
        Read data for dataset SIMF.

        Parameters
        ----------
        subject: str
            Subject name

        *args: list(str)
            Data ids:
            - eeg: EEG (looks for eeg_raw if sample is None)
            - eeg_raw: Coregistered EEG with zeros if exists, else not coregistered EEG with zeros
            - trans: Identity transformation to coregister EEG
            - tracto: Tractography
            - t12diff: Transformation from T1 to diffusion space
            - gt: Ground truth
            For the rest, check read_freesurfer and read_mindmap

        sample: str
            Sample name for EEG.

        mri_dtype: str
            MRI dtype (Nibabel or ANTs) for MRI images.

        Returns:
        data: dict
            Dictionnary of data using data ids as keys.
        """
        if isinstance(subject, int):
            subject = self.config["SUBS"][subject-1]

        name    = self.config["NAME"]
        mm_dir  = self.config["MM_DIR"]
        cmp_dir = self.config["CMP_DIR"]
        sim_dir = self.config["SIM_DIR"]

        data = dict(name=name, sub=subject)

        if isinstance(sample, int):
            sample = f"sample{sample:03}"

        # dataset specifics (use this section to also overwrite freesurfer and mindmap)
        for key in args:
            if key.lower() == "eeg":
                if sample is None:
                    # look for raw EEG
                    d = self.__call__(subject, "eeg-raw")
                    data[key] = d["eeg-raw"]
                else:
                    file = sim_dir/"eeg"/f"{subject}-{sample}_ave.fif"
                    data[key] = mne.Evoked(file, verbose="warning")

            if key.lower() == "eeg-raw":
                file = mm_dir/subject/"eeg"/f"{subject}_coreg-ave.fif"

                if file.exists():
                    print("\033[93mCoregistered EEG found\033[0m")
                    data[key] = mne.Evoked(file, verbose="warning")

                else:
                    print("\033[93mNo coregistered EEG found, making EEG from scratch\033[0m")
                    # null EEG
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

            if key.lower() == "trans":
                data[key] = mne.transforms.Transform("head", "mri")    # identity transformation

            if key.lower() == "tracto":
                # Tractogram
                if subject == "vepcon_sub-10":
                    file = cmp_dir/subject/"dwi"/f"sub-10_model-CSD_desc-DET_tractogram.trk"
                else:
                    file = cmp_dir/subject/"fibtracking"/"sd_stream"/"dwi_denoised_eddy_distc_corrected_1mm_shore_sdstream.trk"
                data[key] = nib.streamlines.load(file)

            if key.lower() == "t12diff":
                # Transforms from T1 to diffusion
                if subject == "vepcon_sub-10":
                    file = cmp_dir/"vepcon_sub-10"/"xfm"/"final0GenericAffine.mat"
                else:
                    file = cmp_dir/subject/"transforms"/"all_t12diff0GenericAffine.mat"
                affine = sp.io.loadmat(str(file))["AffineTransform_double_3_3"].reshape(3, 4, order="F")
                data[key] = np.concatenate((affine, np.array([[0, 0, 0, 1]])))


            if key.lower() == "diff-affine":
                # Affine (diffusion -> T1)
                file   = cmp_dir/subject/"xfm"/"final0GenericAffine.mat"
                affine = sp.io.loadmat(str(file))["AffineTransform_double_3_3"].reshape(3, 4, order="F")
                data[key] = np.concatenate((affine, np.array([[0, 0, 0, 1]])))

            if key.lower() == "gt":
                # ground truth 
                if sample is None: raise ValueError("Require sample name to read data 'gt'.")
                file = sim_dir/"stc"/f"{subject}-{sample}_stc.fif"
                data[key] = mne.read_source_estimate(file) 

            if key.lower() == "lam":
                # eigenvalues
                file = mm_dir/subject/"sc"/f"{subject}_lams-template.npy"
                data[key] = np.load(file)

        args = list(set(args) - set(data.keys()))   # update list of missing arguments

        # FreeSurfer
        d    = self.read_freesurfer(subject, *args, mri_dtype=mri_dtype)     
        data.update(d)                                                 
        args = list(set(args) - set(data.keys()))                   

        # mindmap
        d    = self.read_mindmap(subject, *args, sample=sample)
        data.update(d)
        args = list(set(args) - set(data.keys()))

        if verbose == 1:
            self.log(data, args)

        return data