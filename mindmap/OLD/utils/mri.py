from pathlib import Path
import nibabel as nib
from typing import Optional, List, Tuple, Dict, Callable
import numpy as np
import pandas as pd
from ..io.readers import read_clut, _retrieve_env
from .misc import get_folders
import subprocess
import ants
import mne
import mindmap as mm
import pyvista as pv
import os


# -----------------------------------------------------------------------------------
# 
# MRI parcellation.
# ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
# The Parcellation class is designed to manage brain parcellations. It supports basic 
# operations such as adding two parcellations together and deleting specific regions. 
# Additionally, it allows for region renaming by adding suffixes, changing 
# capitalization, or assigning completely new names.
#
# make_dparc generates a deterministic brain parcellation using FreeSurfer.
#
# make_pparc generates a probability volume parcellation by registering a
# probability volume atlas to the subject.

# -----------------------------------------------------------------------------------

class Parcellation():
    """
    Class to process parcellations.

    Attributes
    ----------
    n_regions: int
        Number of regions

    volume: 3D np.Array
        Parcellation volume

    lut: dict[str, Any]
        Color Lookup Table
        - #No.:int = id of label
        - Label name: str = label name
        - R: int, G: int, B: int, A: float: RGBA color code

    affine: 4x4 np.Array
        Affine transformation to go from voxel space to mri space.
    
    Methods
    -------
    __init__(parc_fname=None, clut_fname=None)
        Initialize parcellation. Load the parcellation if file name is provided.

    add(new_parc, cmap)
        Combines a new parcellation to the current parcellation. The new parcellation
        will overwrite previous regions if they overlap.

    get_nifti()
        Returns a nifti image of the parcellation.

    get_image()
        Returns a 4D np.Array of the volume. The forth dimension contains the rgb
        values for each voxel.

    get_clut()
        Returns the color lookup table as a pandas DataFrame.

    retain(regions)
        Retains regions of interest only.

    save(parc_fname, clut_fname=None)
        Save the parcellation in memory.

    load(parc_fname, clut_fname=None)
        Load the parcellation from memory.
    """

    def __init__(
            self, 
            volume: Optional[str | Path | nib.Nifti1Image] = None, 
            clut: Optional[str | Path | pd.DataFrame] = None
    ) -> None:
        """
        Initialize parcellation.

        Parameters
        ----------
        volume: Optional[path-like | nib.Nifti1Image]
            Volume of parcellation or path to volume.

        clut_fname: Optional[path-like]
            Color lookup table.
        """

        if volume is None:
            self.n_regions = 0 
            self.volume = None 
            self.affine = None 
        elif isinstance(volume, str) or isinstance(volume, Path):
            self.load(volume_fname=volume)
        else:
            self.volume = volume.get_fdata()
            self.affine = volume.affine 
            self.n_regions = len(np.unique(self.volume)) - 1        # -1 to not count 0


        if clut is None:
            self.clut = {"#No.":[], "Label name":[], "R":[], "G":[], "B":[], "A":[]}
        elif isinstance(clut, str) or isinstance(clut, Path):
            self.load(clut_fname=clut)
        else:
            self.clut = clut.to_dict(orient="list")

        self.info = pd.DataFrame({
            "Region ID":self.clut["#No."], 
            "Region name": self.clut["Label name"],
            "RGBA": [(self.clut["R"][i], self.clut["G"][i], self.clut["B"][i], self.clut["A"][i]) for i in range(len(self.clut["R"]))],
            })

    # load & save
    # -----------
            
    def load(
            self,
            volume_fname: str | Path = None,
            clut_fname: str | Path = None,
    ) -> None:
        """
        Load parcellation, or color lookup table, or both.

        Parameters
        ----------
        parc_fname: path-like
            Parcellation file name.

        clut_fname: path-like.
            Color lut file name.
        """

        if volume_fname is not None:
            img = nib.load(volume_fname)
            self.volume = img.get_fdata().astype(int)
            self.affine = img.affine
            self.n_regions = len(np.unique(self.volume)) -1     # -1 to not count 0

        if clut_fname is not None: 
            clut = read_clut(clut_fname)
            self.clut = clut.to_dict(orient="list")


    def save(
            self,
            parc_fname: str | Path,
            clut_fname: Optional[str | Path] = None,
    ) -> None:
        """
        save image as nifti image (.nii) and color look up table as text (.txt).
        
        Parameters
        ----------
        parc_fname: path-like
            Parcellation file name.

        clut_fname: Optional[path-like]
            Color lut file name.
        """

        # save image
        img = self.get_nifti()
        nib.save(img, parc_fname)

        # save lookup table
        if clut_fname:
            lut = self.get_clut()
            
            lines = ["# blablabla \n"]

            make_line = lambda row: f"{row['#No.']:<4} {row['Label name']:<60} {row['R']:>3} {row['G']:>3} {row['B']:>3} {row['A']:>3}"
            lines += [make_line(row) for _, row in lut.iterrows()]

            with open(clut_fname, "w") as colorLUT_f:
                colorLUT_f.write("\n".join(lines))

    def id2name(self, id_labels):
        id_to_name = self.info.set_index("Region ID")["Region name"].to_dict()
        return np.array([id_to_name.get(i, "Unknown") for i in id_labels])
    
    def name2id(self, name_labels):
        name_to_id = self.info.set_index("Region name")["Region ID"].to_dict()
        return np.array([name_to_id.get(name, 0) for name in name_labels])

    # get
    # ---

    def get_nifti(self) -> nib.Nifti1Image:
        """
        Returns the parcellation as nifti image.
        """
        return nib.Nifti1Image(self.volume, self.affine)
    
    def get_clut(self) -> pd.DataFrame:
        """
        Returns the lut (look up table) as pandas DataFrame.
        """
        return pd.DataFrame(self.clut)
    
    def get_image(self) -> np.ndarray: 
        """
        Returns an RGB volumetric matrix (4D) of the parcellation.
        """
        indices = self.clut["#No."]
        rgb = np.array([self.clut["R"], self.clut["G"], self.clut["B"]]).T
        cmap = {i:list(c) for i,c in zip(indices, rgb)}

        # make RGB image
        image = np.empty(self.volume.shape + (3,), dtype=int)
        for value, color in cmap.items():
            indices = self.volume == value 
            image[indices] = color 

        return image
    
    def label_points(self, points: np.ndarray, id: bool = False, cras: np.ndarray = np.zeros(3)) -> List[str]:
        """
        Get labels of the points given as input.

        Parameters
        ----------
        points: np.ndarray | mne.SourceSpaces (Nx3)
            Points of interest.

        id: bool
            Returns label id rather than label name if True.

        Returns
        -------
        labels: list[str]
            List of size N indicating for each point the regions it belongs to.
        """
        if isinstance(points, mne.SourceSpaces):
            src    = points.copy()
            points = mm.utils.get_src_position(points)*1000
        else:
            src = None

        # apply cras
        points += cras

        # convert points [mm] into voxel coordinates using affine
        inv_trans = np.linalg.inv(self.affine)
        voxels    = nib.affines.apply_affine(inv_trans, points)
        voxels    = np.round(voxels).astype(int)

        # retrieve labels at voxel coordinates
        labels = self.volume[voxels[:,0], voxels[:,1], voxels[:,2]].astype(int)

        if src is not None:
            labels = mm.utils.correct_labels(labels, src, self)

        if id == False:
            labels = self.id2name(labels)

        return np.array(labels)
    
    def get_colors(self, labels):
        color_table = np.array([self.clut["R"], self.clut["G"], self.clut["B"]]).T

        if isinstance(labels[0], str):
            label_to_color = dict(zip(self.clut["Label name"], color_table))
            colors         = [label_to_color[l] for l in labels]

        else:
            id_to_color = dict(zip(self.clut["#No."], color_table))
            colors      = [id_to_color[id] for id in labels]

        return np.vstack(colors)
    
    def get_position(self, rois=None):
        """
        Get position of regions of interest. The position is defined as the center of mass of 
        all voxels belonging to the region. 
        """
        if rois is None:
            rois = self.clut["#No."]
        if (not isinstance(rois, list)) and (not isinstance(rois, np.ndarray)):
            rois = [rois]
        if isinstance(rois[0], str):
            rois = self.name2id(rois)

        positions = []

        for r in rois:
            voxels = np.argwhere(self.volume == r)
            pos    = np.mean(voxels, axis=0)
            pos    = nib.affines.apply_affine(self.affine, pos)
        
            positions.append(pos)

        return np.vstack(positions)

    def get_volumes(self, labels: list[str] | list[int] = None) -> list[float]:
        """
        Retrieve volumes per brain region.

        Parameters
        ---------
        labels: list[str] | list[int]
            Label names or ids to retrieve. If None, retrieve all volumes.

        Returns
        -------
        volumes: list[float]
            Volumes in mm³.
        """ 
        # set labels as index for fast lookup
        if labels is not None:
            if isinstance(labels[0], str):
                volume_map = self.info.set_index("Region name")["Volume"]
            else:
                volume_map = self.info.set_index("Region ID")["Volume"]
            volumes = np.array([volume_map.get(l, np.nan) for l in labels])
        
        else:
            volumes = self.info["Volume"].values

        return volumes
        
    
    def create_mask(self, regions: List[int] = None) -> nib.Nifti1Image:
        """
        Create mask of parcellation.

        Parameters
        ----------
        regions: list[int]
            Indices of regions to include in the mask. If 'regions' is None then all
            regions will be included.
        """
        if regions is None:
            regions = self.clut["#No."]
        if (not isinstance(regions, list)) and (not isinstance(regions, np.ndarray)):
            regions = [regions]
        if isinstance(regions[0], str):
            regions = self.name2id(regions)
            
        mask = np.isin(self.volume, regions)

        return nib.Nifti1Image(mask.astype(float), self.affine)
    
    # Basic operations
    # ----------------

    def join(self, other: "Parcellation") -> None:
        """
        Join two parcellations.

        Parameters
        ----------
        other: Parcellation
            Parcellation to join.
        """
        # check if affines and volume shape match between both parcellations
        if self.affine is None:
            self.affine = other.affine
        else:
            
            assert np.all(np.isclose(self.affine, other.affine, atol=1e-6)), "Affines of parcellations don't match, cancel joining of parcellations."

        if self.volume is None:
            self.volume = np.zeros_like(other.volume)
        else:
            assert np.all(self.volume.shape==other.volume.shape), "Volume spaces of parcellations don't match, cancel joining of parcellations."

        # join volumes
        self.reset_No()                            # reset indices
        other.reset_No(offset=self.n_regions+1)

        new_volume = self.volume.copy()     
        mask = other.volume > 0
        new_volume[mask] = other.volume[mask]      # non-zero values of 'other' will overwrite values of 'self'.
 
        # join cluts
        new_clut = dict()
        for key in self.clut.keys():
            new_clut[key] = self.clut[key] + other.clut[key]

        new_parc = Parcellation(nib.Nifti1Image(new_volume, self.affine), pd.DataFrame(new_clut))
        new_parc.reset_No()

        return new_parc  


    def retain(self, regions: List[Tuple[int] | int]) -> "Parcellation":
        """
        Recreates a parcellation that retains only the regions of interest.

        Parameters
        ----------
        regions: list[tuple[int] | int]
            Regions of interest as id ranges.
            e.g. [(1000, 2000), 16] -> ids from 1000 to 2000 + id == 16

        Returns
        -------
        new_parcellation: Parcellation
            Parcellation containing only the regions of interest.
        """
        # build mask
        ids = []    # to memorize order, sort in this order at the end
        mask = np.zeros_like(self.volume, dtype=bool) 
        for r in regions:
            if isinstance(r, tuple):
                mask += (self.volume >= r[0]) & (self.volume <= r[1])
                ids += list(np.unique(self.volume[mask]))
            else:
                mask += (self.volume == r)
                if (mask.sum()>0): ids += [r]

        # filter volume
        new_volume = np.zeros_like(self.volume)
        new_volume[mask] = self.volume[mask]

        # filter clut
        new_clut = self.get_clut()
        new_clut = new_clut[new_clut["#No."].isin(ids)]
        new_clut.sort_values(by="#No.", key = lambda column: column.map(lambda e: ids.index(e)), inplace=True)

        # generate new parcellation
        image = nib.Nifti1Image(new_volume, self.affine) 

        return Parcellation(image, new_clut)

    # Label change
    # ------------

    def reset_No(self, offset: int = 1) -> None:
        """
        Reset the '#No.' property of the regions. If some '#No.' appear in the clut but
        not in the volume, they will be omitted.

        Parameters
        ----------
        offsent: int
            Offset at which to start the first '#No.'.
        """

        ids = np.unique(self.volume)
        ids = np.delete(ids, 0)             # don't consider 0 voxels
        new_ids = np.arange(len(ids)) + offset 

        # reset volume
        volume_copy = self.volume.copy()
        for old, new in zip(ids, new_ids):
            self.volume[volume_copy==old] = new 

        # reset clut 
        mask = np.isin(self.clut["#No."], ids)
        for key in self.clut.keys():
            self.clut[key] = np.array(self.clut[key])[mask].tolist()      # masks only works on np.array

        self.clut["#No."] = new_ids.tolist()


    def change_labels(
            self, 
            prefix: str = "", 
            lower: bool = False, 
            replace: Dict[str, str] = {}
    ) -> None:
        """
        Change labels.

        Parameters
        ----------
        prefix: str
            Add prefix to labels.

        lower: bool
            Lower all letters.

        replace: dict[str, str]
            Replace pattern with new pattern.
        """

        for old, new in replace.items():
            self.clut["Label name"] = [label.replace(old, new) for label in self.clut["Label name"]]

        if lower == True:
            self.clut["Label name"] = [label.lower() for label in self.clut["Label name"]]

        self.clut["Label name"] = [prefix + label for label in self.clut["Label name"]]

    def change_ids(
            self, 
            ids: list[int | tuple], 
            operator: Callable, 
            new_clut: pd.DataFrame = None
    ) -> None:
        """
        Change values of IDs.
        
        Parameters
        ----------
        ids: list[int]
            ids to change. If an element is a tuple, it represents a range.

        operator: function
            Operation to apply on ids.

        new_clut: pd.DataFrame
            New color look up table for new ids. If None, the old color look up table
            is kept.
        """
        # create mask
        mask = np.zeros_like(self.volume, dtype=bool)
        for id in ids:
            if isinstance(id, tuple):
                mask += (self.volume >= id[0]) & (self.volume <= id[1])
            else:
                mask += (self.volume == id)

        # apply operator
        self.volume[mask] = operator(self.volume[mask])


    # format conversion
    # -----------------

    def prob2dseg(self, thresh: float, mask: nib.Nifti1Image = None) -> "Parcellation":
        """
        Convert a probabilistic parcellation to a deterministic parcellation.

        Parameters
        ----------
        thresh: float
            Threshold for probabilities to be considered.

        mask: nib.Nifti1Image
            Restricted region to define the segmentation.
        """
        # check inputs
        volume = self.volume.copy()
        assert len(volume.shape) == 4, "This parcelation is not a probabilistic parcellation (requires 4D volume). Cancel prob2dseg."

        volume[volume<thresh] = 0
        volume[volume>1] = 1            # Is it ever bigger than 1?

        #  probabilistic to deterministic
        idx = np.where(np.sum(volume, axis=3)==0)       # voxels with 0 probability of being thalamus nuclei
        new_volume = volume.argmax(axis=3) + 1          # assign id according to max probability
        new_volume[idx] = 0                             # replace non thalamic voxels with 0

        # apply mask
        if mask is not None:
            mask = mask.get_fdata().astype(bool)
            new_volume[~mask] = 0         

        image = nib.Nifti1Image(new_volume.astype(float), self.affine)
        parcellation = Parcellation(image, self.get_clut())

        return parcellation

    def compute_volumes(self):
        """
        Compute volume in [mm³] of each brain region.
        """
        # compute volume per region
        voxels_per_region = np.bincount(self.volume.ravel())
        voxel_volume      = extract_voxel_volume(self.affine)
        volume_per_region = voxel_volume * voxels_per_region

        # keep only existing regions
        self.info["Volume"] = volume_per_region[self.info["Region ID"]]


def correct_labels(labels, sources, parcel):
    """
    Correct labels that misplace left and right hemisphere dipoles because the parcellation is not perfect.
    """
    # map to switch from label id to label name
    clut = parcel.get_clut()
    map  = pd.Series(clut["Label name"].values, index=clut["#No."]).to_dict()
    
    label_names = [map[id] for id in labels]

    # left/right hemisphere seperation
    split = len(sources[0]["vertno"])

    # switch 'lh' labels in right hemisphere to 'rh' homologue and vice virsa
    label_names[:split] = [l.replace("-rh-", "-lh-") for l in label_names[:split]]
    label_names[split:] = [l.replace("-lh-", "-rh-") for l in label_names[split:]]

    # convert back to lable ids
    new_labels = [list(map.keys())[list(map.values()).index(n)] for n in label_names]

    return new_labels


def parcelate(sub, parcellation="multiconn", margin:float = 0, threshold:float = 0.05, verbose:int = 1):
    """
    Parcelate subject according to given parcellation.

    Parameters:
    ----------
    sub: str
        Subject name.

    parcellation: str
        Parcellation name.
        - multiconn
        - Lausanne2018
        - Thalamic

    margin: float
        Margin for the Lausanne 2018 atlas.

    threshold: float:
        Probability threshold for the Thalamic atlas.
        
    Returns:
    -------
    parcel: mm.Parcellation
        Parcellation
    """

    if parcellation.lower() == "lausanne2018":
        parcel = parcelate_lausanne2018(sub, margin=margin, verbose=verbose)
    elif parcellation.lower() == "thal":
        parcel = parcelate_thal(sub, threshold=threshold, verbose=verbose)
    elif parcellation.lower() == "multiconn":
        parcel = parcelate_multiconn(sub, margin, threshold, verbose=verbose)
    else:
        raise ValueError(f"Unknown Parcellation '{parcellation}'")
        
    return parcel


def parcelate_lausanne2018(sub:str, margin:float = 0, verbose:int = 1):
    """
    Register the Lausanne 2018 atlas to subject. Output in FreeSurfer space.

    Parameters:
    ----------
    sub: str
        Subject

    margin: float
        Margin of gray matter area onto white matter area

    verbose: int
        Verbosity level (0 or 1)

    Returns:
    -------
    laus2018: mm.Parcellation
        Lausanne2018 Parcellation in FreeSurfer space.
    """
    if verbose == 1:
        print(f"\033[96mSegmenting subject {sub} according to the Lausanne 2018 atlas ...\033[0m")

    env = _retrieve_env("FS_DIR", "FS_HOME")

    # retrieve data
    if verbose == 1:
        print(f"\033[96mLoading data ...\033[0m")
    std  = mm.io.read_standards("ctx")

    # set freesurfer configuration
    os.environ["SUBJECTS_DIR"]    = str(env["FS_DIR"])
    os.environ["FREESURFER_HOME"] = str(env["FS_HOME"])  

    # temporary files
    annotL    = env["FS_DIR"]/sub/"label"/"lh.my-atlas.annot" 
    annotR    = env["FS_DIR"]/sub/"label"/"rh.my-atlas.annot" 
    dparc_fs  = env["FS_DIR"]/sub/"mri"/"my-atlas_space-fs_dparc.nii.gz"

    # Registration of surface-based segmentation
    if verbose == 1:
        print(f"\033[96mRegistering surface-based segmentation ...\033[0m")
    subprocess.run(
        [
            "mri_surf2surf",
            "--srcsubject", "fsaverage",
            "--trgsubject", sub,
            "--hemi", "lh",
            "--sval-annot", std["ctx"]["LAUSL"],
            "--tval", str(annotL),
        ],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    subprocess.run(
        [
            "mri_surf2surf",
            "--srcsubject", "fsaverage",
            "--trgsubject", sub,
            "--hemi", "rh",
            "--sval-annot", std["ctx"]["LAUSR"],
            "--tval", str(annotR),
        ],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    # Expanding surface-based segmentation to volumetric segmentation
    if verbose == 1:
        print(f"\033[96mExpanding surface-based segmentation to volumetric segmentation ...\033[0m")
    subprocess.run(
        [
            "mri_aparc2aseg",
            "--s", sub,
            "--annot", "my-atlas",
            "--wmparc-dmax", str(margin),
            "--labelwm",
            "--hypo-as-wm",
            "--new-ribbon",
            "--o", str(dparc_fs),
        ],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    # load data
    volume  = nib.load(dparc_fs)
    fs_clut = mm.io.read_clut(env["FS_HOME"]/"FreeSurferColorLUT.txt")     # FreeSurfer color LUT 
    lh_clut = mm.io.read_clut(annotL)                                      # Left annotation color LUT
    rh_clut = mm.io.read_clut(annotR)                                      # Right annotation color LUT 

    # correct clut
    if verbose == 1:
        print(f"\033[96mFixing FreeSurfer color LUT ...\033[0m")
    laus2018_clut = fs_clut[(fs_clut["#No."]<1000) | (fs_clut["#No."]>4999)]     # Keep non-cortical regions 

    gm_lh_clut, wm_lh_clut = lh_clut.copy(), lh_clut.copy()
    gm_rh_clut, wm_rh_clut = rh_clut.copy(), rh_clut.copy()

    gm_lh_clut["#No."] += 1000          # 1000-1999 
    gm_rh_clut["#No."] += 2000          # 2000-2999
    wm_lh_clut["#No."] += 3000          # 3000-3999
    wm_rh_clut["#No."] += 4000          # 4000-4999

    laus2018_clut = pd.concat([laus2018_clut, gm_lh_clut, gm_rh_clut, wm_lh_clut, wm_lh_clut])
    laus2018_clut = laus2018_clut.sort_values(by=["#No."])

    laus2018 = Parcellation(volume, laus2018_clut)
    laus2018.change_ids([(3001, 3999), (4001, 4999)], lambda x: x-2000)

    # delete temporary files
    if verbose == 1:
        print(f"\033[96mDeleting Temporary files ...\033[0m")
    annotL.unlink()
    annotR.unlink()
    dparc_fs.unlink()

    if verbose == 1:
        print(f"\033[92mLausanne Parcellation of subject {sub} finished\033[0m]")

    return laus2018

    

def parcelate_thal(sub, thresh=0.05, verbose=1):
    """
    Register the Thalamic atlas to the subject.

    Parameters:
    ----------
    sub: str
        Subject's name.

    thresh: float
        Threshold to convert probabilistic atlas to deterministic atlas.
    
    verbose: int
        Verbosity level.

    Returns:
    -------
    thal: mm.Parcellation
        Thalamic parcellation in FreeSurfer space.
    """
    if verbose == 1:
        print(f"\033[96mSegmenting subject {sub} according to the thalamic atlas ...\033[0m")

    # retrieve data
    if verbose == 1:
        print(f"\033[96mLoading data ...\033[0m")
    data = mm.io.read_dataset(sub, "T1", mri_dtype="ants")
    std  = mm.io.read_standards("mni", "thal", "thal_clut")

    if verbose == 1:
        print(f"\033[96mANTS registration of MNI space ...\033[0m")
    # registration transform
    mytx = ants.registration(
        fixed = std["mni"],
        moving = data["T1"],
        type_of_transform = "SyN",
    )

    # registration 
    volume = ants.apply_transforms(
        fixed = data["T1"],
        moving = std["thal"],
        transformlist = mytx["invtransforms"],
        imagetype = 3,
        interpolator = "linear",
    )

    volume = mm.utils.ants_to_nibabel(volume)
    thal   = Parcellation(volume, std["thal_clut"])
    thal   = thal.prob2dseg(thresh=thresh)

    if verbose == 1:
        print(f"\033[92mThalamic Parcellation of subject {sub} finished\033[0m]")

    return thal

def parcelate_multiconn(sub: str, margin:float = 2, thresh:float = 0.05, verbose=1):
    """
    Generate the multiconn parcellation for subject.

    Parameters:
    ----------
    sub: str
        Subject

    margin: float
        Margin for Lausanne 2018 atlas

    thresh: float
        Probability threshold for Thalamic atlas.

    Returns:
    -------
    multi_conn: mm.Parcellation
        Multi conn parcellation in FreeSurfer space.
    """
    # ***** Generate Lausanne2018 and Thalamic parcellations *****

    # Lausanne Parcellation
    laus2018 = parcelate_lausanne2018(sub, margin=margin, verbose=verbose)

    # Thal Parcellation
    thal = parcelate_thal(sub, thresh=thresh, verbose=verbose)

    # ***** Fuse Lausanne2018 and Thalamic parcellations *****

    # Extract and rename parts of interest from the Lausanne 2018 parcellation.
    ctx_lh = laus2018.retain([(1001, 1999), (3001, 3999)])
    ctx_lh.change_labels(prefix="ctx-lh-")

    ctx_rh = laus2018.retain([(2001, 2999), (4001, 4999)])
    ctx_rh.change_labels(prefix="ctx-rh-")

    subc_lh = laus2018.retain([11, 12, 13, 26, 18, 17])
    subc_lh.change_labels(replace={"Left-":"", "Accumbens-area":"accumbens_area"}, prefix="subc-lh-", lower=True)

    subc_rh = laus2018.retain([50, 51, 52, 58, 54, 53])
    subc_rh.change_labels(replace={"Right-":"", "Accumbens-area":"accumbens_area"}, prefix="subc-rh-", lower=True)

    # Extract and rename parts of interest from the thalamic parcellation.
    thal_lh = thal.retain([(8, 14)])

    thal_rh = thal.retain([(1,7)])

    brain_stem = laus2018.retain([16])
    brain_stem.change_labels(replace={"-":"_"})

    # join parts of interest
    multi_conn = mm.utils.Parcellation()
    multi_conn = multi_conn.join(ctx_rh)
    multi_conn = multi_conn.join(thal_rh)
    multi_conn = multi_conn.join(subc_rh)
    multi_conn = multi_conn.join(ctx_lh)
    multi_conn = multi_conn.join(thal_lh)
    multi_conn = multi_conn.join(subc_lh)
    multi_conn = multi_conn.join(brain_stem)

    if verbose == 1:
        print(f"\033[92mMultiConn parcellation of subject {sub} finished\033[0m]")

    return multi_conn


# -----------------------------------------------------------------------------------
# 
# MRI head model.
# ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
# There are two main ways of making individual head models. Using a boundary element
# model (BEM) or a finite element model (FEM). In the BEM, one represents the surfaces
# of the different layers between the sources and the electrodes. These mainly include
# the brain surface, the inner skull, the outer skull and the inner skin. Regions 
# between these surfaces are considered homogeneous (only one type of tissue). In the 
# FEM, every voxel is labeled with a tissue type.
#
# For this project we implement a BEM as it is more simple to start with.
# After the FreeSurfer default pipeline we already have the brain surface. The remaining
# surfaces can be obtained using the watershed algorithm implemented in FreeSurfer.
#
# make_bem() calls the watershed algorithm and saves all surfaces in a 'bem' folder 
# located in the subject's FreeSurfer directory.
# The watershed algorithm is not optimal and often generate errors. We therefore also 
# provided an automatic correction using correct_skull(). This function will detect 
# inner skull nodes that are outside the outer skull nodes and move them back inside.
# This will not correct for eventual other errors generated with watershed.
# -----------------------------------------------------------------------------------

def make_bem(
        subject: str,
        conductivities: list[float],
        min_skull_thick: float,
        volume: str = "T1",
        fs_dir: str | Path = None,
        overwrite: bool = False,
        keep_all: bool = False,
) -> mne.bem.ConductorModel:
    """
    Parameters
    ----------
    subjet: str
        FreeSurfer subject name.

    conductivities: list[float]
        Electrical conductivities for the brain, skull and skin.

    min_skull_thick: float
        Minimum skull thickness in [mm]

    volume: str
        FreeSurfer volume to use.

    fs_dir: path-like
        FreeSurfer directory. Will check in os.environ if None.

    overwrite: bool
        If True then overwrite existing bem files.

    keep_all: bool
        If True then all intermediate files are saved.

    Returns
    -------
    bem: mne.bem.ConductorModel
        BEM head model.

    Saves:
    -----
    Files that will be saved if keep_all=True

    [fs_dir]/[subject]/bem/brain.surf
        Mesh of the brain.

    [fs_dir]/[subject]/bem/inner_skull.surf
        Mesh of the inner skull after correcting it.

    [fs_dir]/[subject]/bem/inner_skull_raw.surf
        Mesh of the inner skull before correcting it.

    [fs_dir]/[subject]/bem/outer_skull.surf
        Mesh of the outer skull.

    [fs_dir]/[subject]/bem/outer_skin.surf
        Mesh of the outer skin.

    [fs_dir]/[subject]/bem/inner_skull_changes.vtk
        Vector field showing the displacements of inner skull vertices during the correction.

    [fs_dir]/[subject]/bem/watershed/[subject]_brain_surface
        ???

    [fs_dir]/[subject]/bem/watershed/[subject]_inner_skull_surface
        ???

    [fs_dir]/[subject]/bem/watershed/[subject]_outer_skull_surface
        ???

    [fs_dir]/[subject]/bem/watershed/[subject]_outer_skin_surface
        ???

    [fs_dir]/[subject]/bem/watershed/ws.mgz
        Brain extracted volume.
    """
    # files
    fs_dir = get_folders("make_bem", fs_dir=fs_dir)

    bem_fname = fs_dir/subject/"bem"/f"{subject}-bem.fif"
    if (bem_fname.exists()) and (overwrite==False):
        raise FileExistsError(f"BEM solution already exists: {bem_fname.name}")
    
    bem_folder = fs_dir/subject/"bem"
    if not bem_folder.exists():
        bem_folder.mkdir()

    # create surfaces from MRI
    print("\033[94m \nMAKE_BEM 1/3: creating surfaces \033[0m")
    mne.bem.make_watershed_bem(
        subject=subject,
        subjects_dir=fs_dir,
        volume=volume,
        overwrite=True,
    )

    # correct skull layer
    print("\033[94m \nMAKE_BEM 2/3: correcting surfaces \033[0m")
    correct_skull(subject, min_skull_thick, fs_dir, keep_all)

    # create bem model
    print("\033[94m \nMAKE_BEM 3/3: creating model \033[0m")
    model = mne.make_bem_model(
        subject=subject,
        subjects_dir=fs_dir,
        conductivity=conductivities,
    )

    # handle intermediate files
    if keep_all == False:   
        print("\033[94m \nMAKE_BEM END: Deleting intermediate files \033[0m")
        files = [
            fs_dir/subject/"bem"/"brain.surf",
            fs_dir/subject/"bem"/"outer_skull.surf",
            fs_dir/subject/"bem"/"outer_skin.surf",
            fs_dir/subject/"bem"/"inner_skull.surf",
            fs_dir/subject/"bem"/"inner_skull_raw.surf",
            fs_dir/subject/"bem"/"inner_skull_changes.vtk",
            fs_dir/subject/"bem"/f"{subject}-head.fif",
            fs_dir/subject/"bem"/"watershed"/f"{subject}_brain_surface",
            fs_dir/subject/"bem"/"watershed"/f"{subject}_inner_skull_surface",
            fs_dir/subject/"bem"/"watershed"/f"{subject}_outer_skin_surface",
            fs_dir/subject/"bem"/"watershed"/f"{subject}_outer_skull_surface",
            fs_dir/subject/"bem"/"watershed"/"ws.mgz",
        ]

        for f in files:
            f.unlink(missing_ok=True)
        
        watershed_folder = fs_dir/subject/"bem"/"watershed"
        watershed_folder.rmdir()

    return model 


def correct_skull(
        subject: str,
        offset: float, 
        fs_dir: str | Path,
        save_arrows: bool = False,
) -> None:
    """
    Correct skull by displacing wrong vertices of inner skull back into the outer skull
    with an additional offset. 

    subject: str
        FreeSurfer subject name.

    offset: float
        Minimal distance between inner and outer skull.

    fs_dir: path-like
        FreeSurfer directory.

    save_arrows: bool
        If True then the vector field showing the displacement done.
    """

    # files
    fs_dir = Path(fs_dir)

    inner_fname = fs_dir/subject/"bem"/"inner_skull.surf"
    outer_fname = fs_dir/subject/"bem"/"outer_skull.surf"

    # load
    vertices, faces, metadata = nib.freesurfer.io.read_geometry(inner_fname, read_metadata=True)
    in_skull = mm.utils.surf_to_polydata(vertices, faces)

    vertices, faces = nib.freesurfer.io.read_geometry(outer_fname)
    out_skull = mm.utils.surf_to_polydata(vertices, faces)

    # apply correction
    outputs = surf_correction(
        in_skull, 
        out_skull,
        offset = offset,
        return_arrows = save_arrows,
    )

    # rename old in_skull, save new in_skull as inner_skull.surf
    inner_fname.rename(fs_dir/subject/"bem"/"inner_skull_raw.surf")

    if save_arrows == True: 
        in_skull_fixed, arrows = outputs 
        arrows_fname = fs_dir/subject/"bem"/"inner_skull_changes.vtk"
        arrows.save(arrows_fname)

    else:
        in_skull_fixed = outputs 

    fname = fs_dir/subject/"bem"/"inner_skull.surf"
    vertices = in_skull_fixed.points
    faces = in_skull_fixed.faces

    # rewrite faces into right format
    idx = np.arange(0, len(faces), step=4)
    faces = np.delete(faces, idx).reshape(-1, 3)

    nib.freesurfer.io.write_geometry(fname, vertices, faces, volume_info=metadata)


def run_charm(subID:str, T1: str | Path, T2: str | Path=None, overwrite=False):
    """
    Run the charm command line. The charm comand line generates a head segmentation from a T1 image (optional T2
    for better reconstruction)

    Parameters
    ----------
    folder: Path | str
        Folder in which to generate the output.

    subID: str
        Subject ID.

    T1: Path | str
        Path to T1 image.

    T2: Path | str
        Path to T2 image.

    Files
    ------
    m2m_subID: folder
        Folder containing charm segmentation.
        - m2m_subId/segmentation/labeling.nii.gz: Segemented image.
        - m2m_subId/segmentation/labeling_LUT.txt: CLUT of image.
        - ...
    """
    cleanT1 = False 
    cleanT2 = False

    # check T1
    if isinstance(T1, str):
        T1 = Path(T1)

    if not T1.exists():
        raise FileNotFoundError("T1 not found")

    output = T1.parent/f"m2m_{subID}"/"segmentation"/"labeling.nii.gz"
    if output.exists() and not overwrite:
        raise FileExistsError(f"\033[93mSubject {subID} already has a segmentation.\033[0m")

    folder = T1.parent
    if T1.suffix == ".mgz":
        cleanT1  = True
        T1_input = T1.name.replace(".mgz", ".nii.gz")
        out      = subprocess.run(["mri_convert", T1.name, T1_input], cwd=str(folder))
        out.check_returncode()
        
    else:
        T1_input = T1.name

    # check T2
    if T2:
        if isinstance(T2, str):
            T2 = Path(T2)

        if not T2.exists():
            raise FileNotFoundError("T2 not found")

        folderT2 = T2.parent
        if folderT2 != folder:
            raise ValueError("Parent folder of T1 and T2 must be the same to run charm.")

        if T2.suffix == ".mgz":
            cleanT2  = True
            T2_input = T2.name.replace(".mgz", ".nii.gz")
            out      = subprocess.run(["mri_convert", T2.name, T2_input], cwd=str(folder)) 
            out.check_returncode()
        else:
            T2_input = T2.name

        cml = ["charm", subID, T1_input, T2_input, "--segment", "--initatlas"]  # command line with T2
    else:
        cml = ["charm", subID, T1_input, "--segment", "--initatlas"]            # command line without T2

    # run charm
    out = subprocess.run(cml, cwd=str(folder))

    # clean nii files if original files were mgz
    if cleanT1:
        T1_input = folder/T1_input 
        T1_input.unlink()

    if cleanT2:
        T2_input = folder/T2_input
        T2_input.unlink()

    return


def surf_correction(
        in_skull: pv.PolyData, 
        out_skull: pv.PolyData,
        offset: float,
        return_arrows: bool = False,
) -> tuple[pv.PolyData]:
    """
    """

    # Step 1: Select points of in_skull outside of out_skull
    error_skull = in_skull.boolean_difference(out_skull)
    error_mask  = np.zeros(in_skull.n_points, dtype=bool)

    for point in error_skull.points:
        distance = np.linalg.norm(in_skull.points - point, axis=1) 
        idx = np.where(distance < 10e-8)[0]

        # omit new points created by boolean_difference
        if len(idx) == 0:
            pass 

        # save index of points present in in_skull 
        elif len(idx) == 1:
            error_mask[idx[0]] = 1

        # multiple points in in_skull that coincide should not happen
        else:
            raise ValueError("Multiple points for in_skull.")
        
    # Step 2: Select points of in_skull having distance smaller than offset to out_skull
    
    # compute directions
    in_skull.compute_normals()
    signs = np.where(error_mask, -1, 1)
    signs = np.expand_dims(signs, axis=1)
    directions = in_skull.point_normals*signs 

    # compute ray tracing 
    points, rays, _ = out_skull.multi_ray_trace(in_skull.points, directions)

    # compute distances to out_skull
    distance2surface = 1000 * np.ones(in_skull.n_points)
    for i in range(rays.max()+1):
        idx = np.where(rays==i)
        p = points[idx]

        distance2surface[i] = np.linalg.norm(p-in_skull.points[i], axis=1).min()

    closeness_mask = (distance2surface < offset)

    # Step 3: Correct the inner skull

    final_mask = np.ma.mask_or(error_mask, closeness_mask, shrink=False)
    in_skull_fixed = in_skull.copy(deep=True)

    # arrange arrays for computation
    final_mask = np.expand_dims(final_mask, axis=1)
    distance2surface = np.expand_dims(distance2surface, axis=1)
    
    # change points
    changes = -1 * final_mask * in_skull.point_normals * (offset - signs*distance2surface)
    in_skull_fixed.points = in_skull.points + changes

    
    if return_arrows == True:
        changed_mesh = pv.PolyData(in_skull.points)
        changed_mesh["vectors"] = np.array(changes)
        arrows = changed_mesh.glyph(orient="vectors", scale=True)

        return in_skull_fixed, arrows
    
    return in_skull_fixed



def make_bem_volumetric_masks(segmentation, closing_radius=20, return_skull=False):
    """
    Generate volumetric masks for BEM layers.

    Parameters
    ----------
    segmentation: mm.Parcellation
        Charm head segmentations.

    closing_radius: int
        Size of closing disk (number of voxels).

    return_skull: bool
        Whether to return skull mask.

    Returns
    -------
    skin: nib.Nifti1Image 
        Skin mask.

    oskull: nib.Nifti1Image 
        Outer skull mask.

    iskull: nib.Nifti1Image 
        Inner skull mask.

    skull: nib.Nifti1Image (optional)
        Skull mask.
    """
    # skin 
    skin = data["seg"].create_mask([511])
    skin = mm.utils.nibabel_to_ants(skin)
    skin = skin.morphology("close", radius=closing_radius)
    skin[:,-10:,:] = 0
    skin[:,skin.shape[1]-11,:] = ndimage.binary_fill_holes(skin[:,skin.shape[1]-11,:].numpy())      # close bottom

    # skull
    skull = data["seg"].create_mask([515])
    skull = mm.utils.nibabel_to_ants(skull)
    skull = skull.morphology("close", radius=5)

    # trim (used to trim inner skull for easy mesh)
    trim = data["seg"].create_mask([501, 507, 509]) # other tissues/air/mucosa
    trim = mm.utils.nibabel_to_ants(trim)
    trim2 = skull.morphology("close", radius=100)

    # outer skull envelope
    oskull = skull.morphology("close", radius=30) 
    oskull[:,-20:,:] = 0

    # inner skull envelope
    iskull = trim2 * (1-skull) * (1-trim)
    iskull = iskull.get_mask()
    iskull = iskull.morphology("close", radius=20)
    iskull[:,-30:,:] = 0

    if return_skull:
        return skin, oskull, iskull, skull 
    else:
        return skin, oskull, iskull


# MRI specific utilities

def extract_voxel_volume(affine):
    """
    Computes voxel volume in [mm³] from the affine.
    """
    # defines borders of voxel (parallelepiped)
    v1 = affine[:3, 0]
    v2 = affine[:3, 1]
    v3 = affine[:3, 2]

    # volume of a parallelepiped
    volume = np.abs(np.cross(v1, v2) @ v3)
    
    return volume


