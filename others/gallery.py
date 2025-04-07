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

            # superior view
            plotter.camera_position = "xy"
            plotter.save_graphic(out/f"{sub}-{samp}_{results}_superior.svg")

            # inferior view
            plotter.camera_position = "yx"
            plotter.camera.roll += 90
            plotter.save_graphic(out/f"{sub}-{samp}_{results}_inferior.svg")

            # ventral view
            plotter.camera_position = "zx"
            plotter.camera.roll += 90
            plotter.save_graphic(out/f"{sub}-{samp}_{results}_ventral.svg")

            # dorsal view
            plotter.camera_position = "xz"
            plotter.save_graphic(out/f"{sub}-{samp}_{results}_dorsal.svg")
            
            # medial left view
            plotter.camera_position = "zy"
            plotter.camera.roll += 90
            plotter.save_graphic(out/f"{sub}-{samp}_{results}_medial-left.svg")

            # medial right view
            plotter.camera_position = "yz"
            plotter.save_graphic(out/f"{sub}-{samp}_{results}_medial-right.svg")

        except Exception as e:
            print(f"\033[91mCould not generate images for subject '{sub}', sample '{samp}' due to {e}")




# %%
