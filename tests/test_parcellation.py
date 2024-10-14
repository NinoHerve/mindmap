import mindmap as mm
import pytest
import numpy as np
import nibabel as nib
import pandas as pd

@pytest.fixture
def sphere_parcel():
    # volume with spheres
    volume_spheres = np.zeros((100,100,100))
    sphere_points = [[20,20,20],[20,80,80],[80,80,80]]
    r = 10

    for i, p in enumerate(sphere_points):
        for x in range(100):
            for y in range(100):
                for z in range(100):
                    if np.sqrt((x-p[0])**2 + (y-p[1])**2 + (z-p[2])**2) <= r:
                        volume_spheres[x,y,z] = i+1

    clut_spheres = pd.DataFrame({
        "#No.": [1,2,3],
        "Label name": ["red", "blue", "yellow"],
        "R": [255,0,255],
        "G": [0,0,250],
        "B": [0,255,205],
        "A": [0.,0.,0.],    
    })

    affine = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0,], [0,0,0,0]])

    return mm.utils.Parcellation(nib.Nifti1Image(volume_spheres, affine), clut_spheres)

@pytest.fixture
def cube_parcel():
    # volume with cubes
    volume_cubes = np.zeros((100,100,100))
    cube_points = [[20,80,80], [80,80,20]]
    c = 10

    # volume with cubes
    for i, p in enumerate(cube_points):
        x_min = max(p[0] - c, 0)
        x_max = min(p[0] + c, 99)
        y_min = max(p[1] - c, 0)
        y_max = min(p[1] + c, 99)
        z_min = max(p[2] - c, 0)
        z_max = min(p[2] + c, 99)

        volume_cubes[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = i+1

    clut_cubes = pd.DataFrame({
        "#No.": [1,2],
        "Label name": ["green", "orange"],
        "R": [0,255],
        "G": [255,165],
        "B": [0,0],
        "A": [0.,0.],    
    })  

    affine = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0,], [0,0,0,0]])

    return mm.utils.Parcellation(nib.Nifti1Image(volume_cubes, affine), clut_cubes)


def test_operators(sphere_parcel, cube_parcel):
    """Check if Parcellation.join and Parcellation.retain works as expected."""
    parc1 = sphere_parcel
    parc2 = cube_parcel

    join_parc = parc1.join(parc2)
    volume = join_parc.get_nifti().get_fdata()
    clut = join_parc.get_clut()

    assert np.unique(volume).tolist() == [0,1,2,3,4], "Join 1 volume incorrect"
    assert list(clut["#No."]) == [1,2,3,4], "Join 1 IDs incorrect"
    assert list(clut["Label name"]) == ["red", "yellow", "green", "orange"], "Join 1 labels incorrect"
    
    join_parc = parc2.join(parc1)
    volume = join_parc.get_nifti().get_fdata()
    clut = join_parc.get_clut()

    assert np.unique(volume).tolist() == [0,1,2,3,4,5], "Join 2 volume incorrect"
    assert list(clut["#No."]) == [1,2,3,4,5], "Join 2 IDs incorrect"
    assert list(clut["Label name"]) == ["green", "orange", "red", "blue", "yellow"], "Join 2 labels incorrect"
    
    retain_parc = join_parc.retain([(1,3), 5])
    volume = retain_parc.get_nifti().get_fdata()
    clut = retain_parc.get_clut()

    assert np.unique(volume).tolist() == [0,1,2,3,5], "Retain volume incorrect"
    assert list(clut["#No."]) == [1,2,3,5], "Retain IDs incorrect"
    assert list(clut["Label name"]) == ["green", "orange", "red", "yellow"], "Retain labels incorrect"


def test_label_changing(cube_parcel):
    parc = cube_parcel

    parc.reset_No(offset=10)
    volume = parc.get_nifti().get_fdata()
    clut = parc.get_clut()

    assert np.unique(volume).tolist() == [0,10,11], "Reset (offset 10) volume incorrect"
    assert list(clut["#No."]) == [10,11], "Reset (offset 10) IDs incorrect"

    parc.reset_No()
    volume = parc.get_nifti().get_fdata()
    clut = parc.get_clut()

    assert np.unique(volume).tolist() == [0,1,2], "Reset volume incorrect"
    assert list(clut["#No."]) == [1,2], "Reset IDs incorrect"

    parc.change_labels(prefix = "COOL")
    clut = parc.get_clut()

    assert list(clut["Label name"]) == ["COOLgreen", "COOLorange"], "Change prefix labels incorrect"

    parc.change_labels(lower=True)
    clut = parc.get_clut()

    assert list(clut["Label name"]) == ["coolgreen", "coolorange"], "Change caps labels incorrect"

    parc.change_labels(replace={"cool":"cool_"})
    clut = parc.get_clut()

    assert list(clut["Label name"]) == ["cool_green", "cool_orange"], "Replace labels incorrect"


