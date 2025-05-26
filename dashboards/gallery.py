# Generate a image gallery for vizualizing results on another platform.

# %% Choose parameters
fconfig = "../configurations/spike.yml"
results = "sloreta-0.01"
resec   = True

#%% imports
import mindmap as mm
from pathlib import Path
import numpy as np

# %%
# configuration
config   = mm.io.read_config(fconfig)
subjects = config["SUBS"]
samples  = config["SAMP"]
out      = Path(config["MM_DIR"])/"gallery"/results


# %%
if not out.exists():
    out.mkdir(parents=True)

for sub in subjects:
    for samp in samples:

        try:
            # data 
            data = mm.io.read_dataset(sub, results, "src", "eeg", "resec", "cras")

            # select 50% rising phase
            _, thalf = mm.utils.eeg_rising_phase(data["eeg"], 0.5)
            time     = thalf/data["eeg"].info["sfreq"]
            signal   = np.abs(data[results].crop(time, time).data.squeeze())

            # make plot
            if "resec" in data.keys() and resec:
                plotter = mm.viz.plot_resection(data["src"], data["resec"], data["cras"], signal, style="vox")
            else:
                plotter = mm.viz.plot_brain_signal(data["src"], signal, cmap="reds")

            plotter.off_screen = True
            plotter.remove_scalar_bar()

            # superior view
            plotter.camera_position = "xy"
            plotter.camera.zoom(1.4)
            plotter.screenshot(out/f"{sub}-{samp}_{results}_superior.png", window_size=(1000,1000), return_img=False)

            # inferior view
            plotter.camera_position = "yx"
            plotter.camera.roll += 90
            plotter.camera.zoom(1.4)
            plotter.screenshot(out/f"{sub}-{samp}_{results}_inferior.png", window_size=(1000,1000), return_img=False)

            # ventral view
            plotter.camera_position = "zx"
            plotter.camera.roll += 90
            plotter.camera.zoom(1.4)
            plotter.screenshot(out/f"{sub}-{samp}_{results}_ventral.png", window_size=(1000,1000), return_img=False)

            # dorsal view
            plotter.camera_position = "xz"
            plotter.screenshot(out/f"{sub}-{samp}_{results}_dorsal.png", window_size=(1000,1000), return_img=False)
            
            # medial left view
            plotter.camera_position = "zy"
            plotter.camera.roll += 90
            plotter.camera.zoom(1.4)
            plotter.screenshot(out/f"{sub}-{samp}_{results}_medial-left.png", window_size=(1000,1000), return_img=False)

            # medial right view
            plotter.camera_position = "yz"
            plotter.camera.zoom(1.4)
            plotter.screenshot(out/f"{sub}-{samp}_{results}_medial-right.png", window_size=(1000,1000), return_img=False)

        except Exception as e:
            print(f"\033[91mCould not generate images for subject '{sub}', sample '{samp}' due to {e}")




# %%
