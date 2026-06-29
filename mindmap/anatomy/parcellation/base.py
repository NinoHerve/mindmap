from typing import Callable

from pathlib import Path

import numpy as np
import pandas as pd

import nibabel as nib

from .io import read_clut

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
        volume: str | Path | nib.Nifti1Image = None,
        clut: str | Path | pd.DataFrame = None
    ):
        """
        Initialize parcellation.

        Parameters
        ----------
        volume: Optional[path-like | nib.Nifti1Image]
            Volume of parcellation or path to volume.

        clut_fname: Optional[path-like | pd.DataFrame]
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
            clut_fname: str | Path = None,
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

            with open(clut_fname, "w", encoding="utf-8") as colorLUT_f:
                colorLUT_f.write("\n".join(lines))

    def id2name(self, id_labels):
        """Change region id to region name"""
        id_to_name = self.info.set_index("Region ID")["Region name"].to_dict()
        return np.array([id_to_name.get(i, "Unknown") for i in id_labels])

    def name2id(self, name_labels):
        """Change region name to region id"""
        name_to_id = self.info.set_index("Region name")["Region ID"].to_dict()
        return np.array([name_to_id.get(name, 0) for name in name_labels])

    # getters
    # --------

    def get_nifti(self) -> nib.Nifti1Image:
        """
        Returns the parcellation as nifti image.
        """
        safe_volume = self.volume.astype(np.int32)
        return nib.Nifti1Image(safe_volume, self.affine)

    def get_clut(self) -> pd.DataFrame:
        """
        Returns the lut (look up table) as pandas DataFrame.
        """
        return pd.DataFrame(self.clut)

    def label_points(
            self,
            points: np.ndarray,
            return_id: bool = False,
    ) -> list[str]:
        """
        Get labels of the points given as input.

        Parameters
        ----------
        points: np.ndarray | mne.SourceSpaces (Nx3)
            Points of interest in [mm].

        return_id: bool
            Returns label id rather than label name if True.

        Returns
        -------
        labels: list[str]
            List of size N indicating for each point the regions it belongs to.
        """
        # convert points [mm] into voxel coordinates using affine
        inv_trans = np.linalg.inv(self.affine)
        voxels    = nib.affines.apply_affine(inv_trans, points)
        voxels    = np.round(voxels).astype(int)

        # retrieve labels at voxel coordinates
        labels = self.volume[voxels[:,0], voxels[:,1], voxels[:,2]].astype(int)

        if not return_id:
            labels = self.id2name(labels)

        return np.array(labels)

    
    # setters
    # -------

    def change_labels(
            self,
            prefix: str = "",
            lower: bool = False,
            replace: dict = None
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
        if replace is None:
            replace = dict()

        for old, new in replace.items():
            self.clut["Label name"] = [label.replace(old, new) for label in self.clut["Label name"]]

        if lower:
            self.clut["Label name"] = [label.lower() for label in self.clut["Label name"]]

        self.clut["Label name"] = [prefix + label for label in self.clut["Label name"]]


    def change_ids(
            self,
            ids: list[int | tuple],
            operator: Callable,
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
        for i in ids:
            if isinstance(i, tuple):
                mask += (self.volume >= i[0]) & (self.volume <= i[1])
            else:
                mask += (self.volume == i)

        # apply operator
        self.volume[mask] = operator(self.volume[mask])


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
        self.reset_no()                            # reset indices
        other.reset_no(offset=self.n_regions+1)

        new_volume = self.volume.copy()     
        mask = other.volume > 0
        new_volume[mask] = other.volume[mask]      # non-zero values of 'other' will overwrite values of 'self'.

        # join cluts
        new_clut = dict()
        for key in self.clut.keys():
            new_clut[key] = self.clut[key] + other.clut[key]

        new_parc = Parcellation(nib.Nifti1Image(new_volume, self.affine), pd.DataFrame(new_clut))
        new_parc.reset_no()

        return new_parc


    def retain(self, regions: list[tuple[int] | int]) -> "Parcellation":
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
                if mask.sum()>0: 
                    ids += [r]

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

    def reset_no(self, offset: int = 1) -> None:
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
