from dash import Dash, html, dcc, callback, Output, Input 
import plotly.express as px 
import pandas as pd 
import mindmap as mm
import os
import base64

# configuration
config  = mm.io.read_config("../configurations/spike.yml")
gal_dir = config["GAL_DIR"]

# files
folders = os.listdir(gal_dir)
FILES   = {f:os.listdir(gal_dir + "/" + f) for f in folders}

def b64_image(image_filename):
    """
    Convert file to base64 text: https://community.plotly.com/t/how-to-embed-images-into-a-dash-app/61839
    Update: use "svg+xml" instead of "png".
    """
    with open(image_filename, "rb") as f:
        image = f.read()
    return "data:image/svg+xml;base64," + base64.b64encode(image).decode("utf-8")

app = Dash()

################### Components ####################
parameters = html.Div(
    children = [
        "Subjects",
        dcc.Dropdown(
            id          = "subs",
            options     = config["SUBS"],
            value       = config["SUBS"][0],
            multi       = True,
            placeholder = "Select subjects",
        ),
        "Views",
        dcc.Dropdown(
            id      = "views",
            options = ["superior", "inferior", "ventral", "dorsal", "medial-left", "medial-right"],
            value   = ["posterior", "medial-left", "medial-right"],
            multi   = True,
        ),
        "Source Imaging methods",
        dcc.Dropdown(
            id      = "methods",
            options = ["sloreta-0.01"],
            value   = ["sloreta-0.01"],
            multi   = True
        ),
        "Image size",
        dcc.Slider(
            id    = "size",
            min   = 1, 
            max   = 100,
            step  = 1,
            value = 30,
            marks = None,
        )
    ]
)

infos = html.Div(
    dcc.Dropdown(
        id      = "sub",
        options = config["SUBS"],
        value   = ["sub-pat02"],
        multi   = True,
    )
)

gallery = html.Div(
    id = "gallery",
)

def create_viz(subs, views, methods, size):
    """
    Create a visualization for a given subject.

    Parameters
    ----------
    sub: str
        Subject id
    
    views: list(str)
        List of views to show. Possible views: [anterior, posterior, ventral, dorsal, medial-left, medial-right]

    methods: list(str)
        Source reconstruction methods to show. [csetX, sloretaX]
    """
    if isinstance(subs, str):
        subs = [subs]

    # list of html.Img for each method
    container = []
    for s in subs:
        sub_container = []
        sub_container.append(html.H2(s)) # subject label

        for m in methods:
            sub_container.append(html.H3(m)) # method label

            for v in views:
                sub_container.append(html.P(v)) # view label
                
                try:
                    file = b64_image(f"{gal_dir}/{m}/{s}-spike_{m}_{v}.svg")
                except FileNotFoundError as e:
                    file = None

                sub_container.append(html.Img(src=file, alt="Not available", style = {"width":"100%"})) # image

        container.append(html.Div(children=sub_container, style = {"display":"inline-block", "border":"black solid", "width":f"{str(size)}%"}))

    container = html.Div(
        children = container,
        style    = {
            "height":"95em",
            "overflow-x":"auto",
            "overflow-y":"auto",
            "white-space":"nowrap",
        },
    )

    return container
######################################################

app.layout = [
    parameters,
    gallery,
]

@callback(
    Output("gallery", "children"),
    Input("subs", "value"),
    Input("views", "value"),
    Input("methods", "value"),
    Input("size", "value"),
)
def update_gallery(subs, views, methods, size):
    return create_viz(subs, views, methods, size)

if __name__ == "__main__":
    app.run(debug=True)