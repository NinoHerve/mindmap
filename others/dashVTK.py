from dash import Dash, html, dcc, Input, Output, State
import dash_vtk
from vtk.util.numpy_support import vtk_to_numpy

import mindmap as mm
import numpy as np

################ Parameters ###################
fconfig = "../configurations/spike.yml"
sub     = 1
samp    = 0
method  = "cset-0.1"
###############################################

################ Create 3D image ##############
def create_vtk_view(sub, hemiL, hemiR):
    """
    Create 3D vtk image.

    Parameters
    ----------
    sub: str
        Subject name (for id naming),

    hemiL: pv.PolyData
        Left hemisphere.

    hemiR: pv.PolyData
        Right hemisphere.

    values: np.ndarray
        Brain values.

    Return
    ------
    vtk_view: dash_vtk.View
        Readable object by dash.
    """
    # properties
    max_signal = np.max([np.max(hemiL["signal"]), np.max(hemiR["signal"])])
    cmap       = "Reds"

    # vtk representations
    vtk_representations = []
    for hemi in [hemiL, hemiR]:
        vtk_representations.append(
            dash_vtk.GeometryRepresentation(
                colorMapPreset = cmap,
                colorDataRange = [0, max_signal],
                children       = [
                    dash_vtk.PolyData(
                        points   = hemi.points.ravel(),
                        polys    = vtk_to_numpy(hemi.GetPolys().GetData()),
                        children = [
                            dash_vtk.PointData([
                                dash_vtk.DataArray(
                                    registration = "setScalars",
                                    name         = "signal",
                                    values       = hemi["signal"],
                                )
                            ])
                        ]
                    )
                ],
            )
        )
    
    # vtk view
    vtk_view = dash_vtk.View(
        pickingModes = ["hover"],
        children     = vtk_representations, 
    )

    return vtk_view

#####################################################

#################### Utilities ######################
def get_data(sub, sample, method):
    """
    Retrieve source space and 50% rising phase.
    """
    data = mm.io.read_dataset(sub, method, "eeg", "src", sample=sample, verbose=0)

    # 50% rising phase
    _, thalf = mm.utils.eeg_rising_phase(data["eeg"], 0.5)
    thalf    = thalf/data["eeg"].info["sfreq"]
    signal   = np.abs(data[method].crop(thalf,thalf).data[:,0])

    return data["src"], signal
######################################################

####################### Layout #######################
# configuration
config    = mm.io.read_config(fconfig)
subjects0 = config["SUBS"]
sample    = config["SAMP"][samp]
init_size = 20

# preload data
SUBJECTS   = subjects0.copy()
components = []
print("loading data ...")
for sub in subjects0:
    try:
        src, signal  = get_data(sub, sample, method)
        hemiL, hemiR = mm.utils.mne_to_polydata(src, signal)
        vtk_view     = create_vtk_view(sub,hemiL, hemiR)
        
        components.append(html.Div(
                id       = sub,
                style    = {"height": f"{init_size}%", "width": f"{init_size}", "display":"none"},
                children = vtk_view,
        ))
    
    except OSError:
        print(f"\033[91mSubject '{sub}' does not have data.\033[0m")
        SUBJECTS.remove(sub)
        continue
print("finished loading data!")


parameters = html.Div(
    style={"height": "9%", "width": "100%", "border":"solid black 2px"},
    children=[
        "Subjects",
        dcc.Dropdown(
            id      = "subjects",
            options = SUBJECTS,
            value   = "sub-pat02",
            multi   = True,
        ),

        "Image sizes",
        dcc.Slider(
            id    = "size",
            min   = 1,
            max   = 100,
            step  = 1,
            value = init_size,
            marks = None,
        )
    ]
)

gallery = html.Div(
    style    = {"height": "90%", "width": "100%", "border":"solid black 2px"},
    children = components,
)

# create app
app = Dash(__name__)

app.layout = html.Div(
    style={"height": "calc(100vh - 16px)"},
    children = [parameters, gallery]
)
########################################################

#################### Call backs ########################

@app.callback(
    [Output(sub, "style") for sub in SUBJECTS],
    Input("subjects", "value"),
    Input("size", "value"),
)
def update_gallery(subs, size):
    if isinstance(subs, str):
        subs = [subs]

    styles = []
    for sub in SUBJECTS:
        if sub in subs:
            style.append({"height": f"{size}%", "width": f"{size}%", "display":"inline-block"})
        else:
            style.append({"height": f"{size}%", "width": f"{size}%", "display":"none"})

    return styles
    




if __name__ == "__main__":
    app.run(debug=True)