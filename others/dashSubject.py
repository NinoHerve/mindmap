from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

import mindmap as mm
from io import BytesIO
import mne
import numpy as np
import dash_vtk
from vtk.util.numpy_support import vtk_to_numpy


###################### Parameters #########################
fconfig         = "../configurations/spike.yml"
sloreta_methods = ["sloreta-0.01"]
cset_methods    = ["cset-0.1"]
###########################################################


################ Create 3D image ##############
def create_vtk_view(sub, hemiL, hemiR, resection=None, show_max=False):
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
                colorDataRange = [0, 1],
                children       = [
                    dash_vtk.PolyData(
                        points   = hemi.points.ravel(),
                        polys    = vtk_to_numpy(hemi.GetPolys().GetData()),
                        children = [
                            dash_vtk.PointData([
                                dash_vtk.DataArray(
                                    registration = "setScalars",
                                    name         = "signal",
                                    values       = 1-(hemi["signal"]/max_signal),   # inverse order for correct colormap
                                )
                            ])
                        ]
                    )
                ],
            )
        )
    
    if resection is not None:
        vtk_representations.append(
            dash_vtk.PointCloudRepresentation(
                xyz      = resection.points.ravel(),
                property = {"color":(0,1,0), "point_size":2},
            )
        )

    if show_max:
        pass
    
    # vtk view
    vtk_view = dash_vtk.View(
        pickingModes = ["hover"],
        background   = [0.135,0.135,0.135],
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


######################## Data #############################
config   = mm.io.read_config(fconfig)
subjects = config["SUBS"]

SUBJECTS = subjects.copy()
for sub in subjects:
    try:
        _ = mm.io.read_dataset(sub, *cset_methods, verbose=0)
    except OSError:
        print(f"\033[91mNo data for subject {sub}\033[0m")
        SUBJECTS.remove(sub)
###########################################################


######################### Layout ##########################
# subject info bubble
sub_info = html.Div(
    style    = {"padding":"50px"},
    children = [
        html.H2("Subject Info card"),
        dcc.Dropdown(
            id      = "subject",
            options = SUBJECTS,
            value   = SUBJECTS[0],
            style   = {"max-width":"200px", "color":"black"}
        ),
        dcc.Checklist(
            id      = "shows",
            options = ["show resection", "show source estimation"],
        )
    ]
)


# eeg data
eeg = html.Div(
    children = [
        dcc.Graph(id="signal")
    ],
)

# Source imaging data
cset = html.Div(
    children = [
        html.Div(
            id    = "csetImg",
            style = {"width":"1200px", "height":"1200px"},
        ),
        dcc.Dropdown(
            id      = "csetMethod",
            options = cset_methods,
            value   = cset_methods[0],
            style   = {"width":"200px", "padding-left":"40%"}
        ),
    ]
)

sloreta = html.Div(
    children = [
        html.Div(
            id = "sloretaImg",
            style= {"width":"1200px", "height":"1200px"},
        )
    ]
)


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div(
    style={"height": "calc(100vh - 16px)"},
    children = [
        dbc.Row(html.H1("Nino's laboratory", style={"text-align":"center", "padding":"100px"})),
        dbc.Row(sub_info),
        dbc.Row([
            dbc.Col(eeg, width=4),
            dbc.Col(cset, width=4),
            dbc.Col(sloreta, width=4),
        ])
    ]
)
###########################################################


######################## Callbacks ########################
@app.callback(
    Output("signal", "figure"),
    Input("subject", "value"),
)
def update_eeg(subject):
    data = mm.io.read_dataset(subject, "eeg", verbose=0)

    topo, t50 = mm.utils.eeg_rising_phase(data["eeg"], 0.5)

    signal = mm.viz.plot_eeg(data["eeg"], title="Interictal spike EEG")
    

    return signal


@app.callback(
    Output("csetImg", "children"),
    Output("sloretaImg", "children"),
    Input("subject", "value"),
    Input("csetMethod", "value"),
    Input("shows", "value"),
)
def update_srcImg(sub, cset_method, shows):
    # resection
    if shows is None:
        resec = None
    elif "show resection" in shows or shows is None:
        data  = mm.io.read_dataset(sub, "resec", "cras", verbose=0)
        resec = mm.utils.nib_to_polydata(data["resec"], data["cras"])
    else:
        resec = None

    # data -> vtk image
    src, signal  = get_data(sub, "spike", cset_method)
    hemiL, hemiR = mm.utils.mne_to_polydata(src, signal)
    cset_vtk     = create_vtk_view(sub,hemiL, hemiR, resec)

    src, signal  = get_data(sub, "spike", sloreta_methods[0])
    hemiL, hemiR = mm.utils.mne_to_polydata(src, signal)
    sloreta_vtk  = create_vtk_view(sub, hemiL, hemiR, resec)
        
    return cset_vtk, sloreta_vtk

###########################################################



if __name__ == "__main__":
    app.run(debug=True, port=8888)