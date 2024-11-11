from pathlib import Path
import nibabel as nib
from typing import Optional, List, Tuple, Dict, Callable
import numpy as np
import pandas as pd
from ..io.readers import read_clut
from .misc import get_folders
import subprocess
import ants
import mne
import mindmap as mm
import pyvista as pv


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
        elif isinstance(clut, str) or isinstance(volume, Path):
            self.load(clut_fname=clut)
        else:
            self.clut = clut.to_dict(orient="list")

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
            self.volume = img.get_fdata()
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
    
    def label_points(self, points: np.ndarray, id: bool = False) -> List[str]:
        """
        Get labels of the points given as input.

        Parameters
        ----------
        points: np.ndarray (Nx3)
            Points of interest.

        id: bool
            Returns label id rather than label name if True.

        Returns
        -------
        labels: list[str]
            List of size N indicating for each point the regions it belongs to.
        """
        inv_trans = np.linalg.inv(self.affine)
        voxels    = nib.affines.apply_affine(inv_trans, points)
        voxels    = np.round(voxels).astype(int)

        labels = self.volume[voxels[:,0], voxels[:,1], voxels[:,2]].astype(int)

        if id == False:
            id_to_name = dict(zip(self.clut["#No."], self.clut["Label name"]))
            labels     = [id_to_name[l] for l in labels]

        return labels
    
    def get_colors(self, labels):
        color_table = np.array([self.clut["R"], self.clut["G"], self.clut["B"]]).T

        if isinstance(labels[0], str):
            label_to_color = dict(zip(self.clut["Label name"], color_table))
            colors         = [label_to_color[l] for l in labels]

        else:
            id_to_color = dict(zip(self.clut["#No."], color_table))
            colors      = [id_to_color[id] for id in labels]

        return np.vstack(colors)
    
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
            assert np.allclose(self.affine, other.affine, atol=1e-15), "Affines of parcellations don't match, cancel joining of parcellations."

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