from dash import Dash, html, dcc, callback, Output, Input 
import plotly.express as px 
import pandas as pd 
import mindmap as mm
import os
import base64


############################ Parameters ############################
fconfig = "../configurations/spike.yml"
methods = ["sloreta-0.01", "cset-0.1"]
####################################################################


############################ Utilities ############################
def b64_image(image_filename):
    """
    Convert file to base64 text: https://community.plotly.com/t/how-to-embed-images-into-a-dash-app/61839
    Update: use "svg+xml" instead of "png".
    """
    with open(image_filename, "rb") as f:
        image = f.read()
    return "data:image/png;base64," + base64.b64encode(image).decode("utf-8")


def create_gallery(subjects, methods, views, size):
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
    if len(views) > 1:
        img_size = "50%"
    else: 
        img_size = "100%"

    img_style = {"display":"inline-block", "width":img_size}

    div_subjects = []
    for sub in subjects:

        div_methods  = []
        for m in methods:
            imgs = [html.Div(IMGS[sub][m][v], style=img_style) for v in views]

            div_methods.append(html.Div(
                style    = {"display":"block", "width": "100%"},
                children = [
                    html.H2(m, style={"margin":"10px"}), 
                    *imgs
                ],
            ))

        div_subjects.append(html.Div(
            style    = {"display":"inline-block", "border":"gray 10px", "border-style": "ridge", "border-radius": "25px", "heigth":"100%", "width": f"{size}%"},
            children = [
                html.H1(sub, style={"text-decoration":"underline", "text-align":"center"}), 
                *div_methods
            ],
        ))

        div = html.Div(
            style    = {"height": "80%", "overflow": "auto"},
            children = div_subjects,
        )

    return div # children for gallery


####################################################################

config    = mm.io.read_config(fconfig)
subjects0 = config["SUBS"]
sample    = "spike"
gal_dir   = config["GAL_DIR"]

init_size = 50
views     = ["superior", "inferior", "ventral", "dorsal", "medial-left", "medial-right"]

# preload data
IMGS     = dict()
SUBJECTS = subjects0.copy()
sample = "spike"

for sub in subjects0:
    IMGS[sub] = dict()
    try: 
        for m in methods:
            IMGS[sub][m] = dict()
            for v in views:
                file = b64_image(f"{gal_dir}/{m}/{sub}-spike_{m}_{v}.png")
                
                img = html.Figure(
                    children = [
                        html.Img(
                            src  = file, 
                            alt  = "Not available", 
                            style={"height":"100%", "width": "100%"},
                        ),
                        html.Figcaption(
                            children = v,
                            style = {"text-align":"center"},
                        ),
                    ]
                )
                IMGS[sub][m][v] = img

    except FileNotFoundError:
        print(f"\033[91mNo data for subject {sub} and method {m}")
        IMGS.pop(sub)
        SUBJECTS.remove(sub)
        continue


####################### Layout #######################

parameters = html.Div(
    children = [
        "Subjects",
        dcc.Dropdown(
            id          = "subs",
            options     = SUBJECTS,
            value       = SUBJECTS[0],
            multi       = True,
            placeholder = "Select subjects",
        ),

        "Views",
        dcc.Dropdown(
            id      = "views",
            options = views,
            value   = ["medial-left", "medial-right"],
            multi   = True,
        ),

        "Source Imaging methods",
        dcc.Dropdown(
            id      = "methods",
            options = ["sloreta-0.01", "cset-0.1"],
            value   = ["sloreta-0.01"],
            multi   = True
        ),

        "Image size",
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
    id = "gallery",
    style={"width":"100%", "height":"80%"},
)


# create application
app = Dash()

app.layout = html.Div(
    children = [
        parameters,
        gallery,
    ]
)

######################################################



@callback(
    Output("gallery", "children"),
    Input("subs", "value"),
    Input("views", "value"),
    Input("methods", "value"),
    Input("size", "value"),
)
def update_gallery(subs, views, methods, size):
    if not subs or not views or not methods:
        return []
    if isinstance(subs, str):
        subs = [subs]
    if isinstance(views, str):
        views = [views]
    if isinstance(methods, str):
        methods = [methods]

    return create_gallery(subs, methods, views, size)

if __name__ == "__main__":
    app.run(debug=True)