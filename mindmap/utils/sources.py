import mne 
import numpy as np
import nibabel as nib
import scipy as sp



def mask_sources(sources: mne.SourceSpaces, mask: nib.Nifti1Image, cras: np.ndarray):
    """
    Mask sources not in mask (usually gray matter mask).

    Parameters
    ----------
    sources: MNE.source_space.SourceSpace
        Sources.

    mask: nib.NIfTi1Image
        Binary mask. If a source is on a voxel with value 0 it will be masked ('inuse'=False).

    cras: np.ndarray (3x1)
        Translation to convert image in talairach conform space to individual space.

    Returns
    -------
    sources: MNE.source_space.SourceSpace
        Sources with updated 'inuse', 'use' and 'vertno' attributes.
    """
    sources = sources.copy()

    for src in sources:
        # select used sources
        inuse = src["inuse"].astype(bool)
        pos   = src["rr"][inuse]*1000                           # source positions in talairach space [mm]
        pos   = pos + cras                                      # source positions in subject space [mm]

        # place sources in voxel space
        inv_affine = np.linalg.inv(mask.affine)
        pos_vox    = mne.transforms.apply_trans(inv_affine, pos)
        pos_vox    = np.round(pos_vox).astype(int)               # source positions in voxel space

        # keep only sources inside GM mask
        volume = mask.get_fdata()
        keep   = volume[pos_vox[:,0], pos_vox[:,1], pos_vox[:,2]].astype(bool)

        new_inuse = np.zeros_like(inuse, dtype=bool)
        new_inuse[inuse] = keep 

        # save new variables in source space    
        nuse   = np.sum(new_inuse)                              # number of used sources
        vertno = np.nonzero(new_inuse)[0]                       # vertex number of used sources

        src["inuse"]  = new_inuse
        src["nuse"]   = nuse
        src["vertno"] = vertno

    return sources



def get_src_position(sources):
    "Get source positions."
    positions = []

    for src in sources:
        positions.append(src["rr"][src["vertno"]])

    return np.concatenate(positions, axis=0)


def get_src_directions(sources):
    "Get source directions."
    positions = []

    for src in sources:
        positions.append(src["nn"][src["vertno"]])

    return np.concatenate(positions, axis=0)



def get_geodist(sources):
    """
    Get matrix showing geodesic distances between sources. Left and right hemispheres are seperated surfaces, 
    hence distances between a left and right source is infinite.
    """

    geodist = []
    for src in sources:
        vertno = src["vertno"]
        geodist.append(src["dist"][vertno][:, vertno])

    return sp.sparse.block_diag(geodist, format="csr")




def add_subcortical_dipoles(sources: mne.SourceSpaces, parcellation, cras: np.ndarray=np.zeros(3)):
    """
    Add subcortical dipoles to source space.

    Parameters
    ----------
    source: mne.SourceSpaces
        Source space (FreeSurfer space).

    parcellation: mm.Parcellation 
        Parcellation (RAS space).

    cras: 
        Translation FreeSurfer -> RAS.

    Returns
    -------
        source: mne.SourceSpaces
            Source space with subcortical dipoles. 
    """
    if np.all(cras == np.zeros(3)):
        print("\033[93mNo cras applied.\033[0m")

    # transpose dipole position to RAS space
    pos = mm.utils.get_src_position(sources) * 1000 + cras

    # label dipoles by brain region
    labels = parcellation.label_points(pos, id=False)

    # compute dipole density per brain region
    regions, dip_per_region = np.unique(labels, return_counts=True)
    parcellation.compute_volumes()
    vol_per_region          = parcellation.get_volumes(regions)
    density_per_region      = dip_per_region / vol_per_region
    density                 = density_per_region.mean()

    # labels of none-cortical regions
    mask   = ~parcellation.info["Region name"].str.contains("ctx", case=False, regex=True)
    subctx = parcellation.info.loc[mask, "Region ID"].values

    # add dipoles in every missing region. Place them using k-means
    volume         = parcellation.volume
    vol_per_region = parcellation.get_volumes(subctx)
    points = []
    for i, region in enumerate(subctx):
        voxels       = np.argwhere(volume == region)
        voxel_coords = (nib.affines.apply_affine(parcellation.affine, voxels) - cras)/1000
        k            = max(int(density*vol_per_region[i]), 1)
        kmeans       = sklearn.cluster.KMeans(n_clusters=k, random_state=0).fit(voxel_coords)
        points.append(kmeans.cluster_centers_)

    pos = {
        "rr": np.concat(points),
        "nn": np.ones((len(points), 3)),
    }

    subctx_src = mne.setup_volume_source_space(pos=pos, verbose="warning")

    return sources + subctx_src