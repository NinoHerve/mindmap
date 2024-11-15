#%% imports
import mindmap as mm
import pywt 
import numpy as np
import mne 
import pandas as pd
from pathlib import Path

env      = mm.io.read_config("../configurations/simf.yml")
easy_dir = Path(env["EASY_DIR"])

#%% Dataset info
sub      = "vepcon_sub-10"      # subject
n_trials = 100                  # number of trials
folder   = easy_dir/sub


#%% simulation parameters
duration       = 1              # duration of simulation
rg_num_dipoles = [20, 100]      # range for the number of active dipoles
rg_s_snr       = [1, 10]        # range for the source snr


#%% Check if dataset already exists
if folder.exists():
    raise FileExistsError(f"An easy simulation dataset already exists at location '{folder}'")
else:
    folder.mkdir(parents=True)

#%% data
data  = mm.io.read_dataset(sub, "eeg_coreg", "fwd_hr", "parc", "cras", "white")
sfreq = data["eeg_coreg"].info["sfreq"]

# vertno (to select dipole and instanciate mne.SourceEstimate)
vertno = [data["fwd_hr"]["src"][0]["vertno"], data["fwd_hr"]["src"][1]["vertno"]]

# dipole distance matrix (to grow dipole patch)
dist = [
    data["fwd_hr"]["src"][0]["dist"][vertno[0]][:,vertno[0]], 
    data["fwd_hr"]["src"][1]["dist"][vertno[1]][:,vertno[1]],
]

# signal (for active dipoles)
wavelet     = pywt.ContinuousWavelet("mexh")
[signal, _] = wavelet.wavefun(level=5, length=int(sfreq*duration))
signal      = signal*10**(-8)       # scale to nA

# function relating source SNR to background variance
Ps = np.sum((signal - signal.mean())**2)/len(signal)         # Average power of signal
a  = 1/(1-(1/len(signal)))  

compute_std = lambda snr: np.sqrt(a*Ps*10**(-snr/10))

#%% generate dataset
for i in range(n_trials):
    # select state 
    n     = np.random.randint(rg_num_dipoles[0], rg_num_dipoles[1])
    s_snr = np.random.uniform(rg_s_snr[0], rg_s_snr[1])

    # generate patch
    hemi            = np.random.choice([0,1])                    # choose hemisphere (left or right)
    starting_dipole = np.random.choice(vertno[hemi])             # choose starting dipole
    starting_idx    = np.argwhere(vertno[hemi]==starting_dipole)[0]
    
    ddist = dist[hemi][starting_idx].todense().squeeze()    # distance between dipoles and selected dipole
    roi   = np.argsort(ddist)[:n]                           # choose n closest dipoles (includes selected dipole)
    
    if hemi == 1: 
        roi = roi + data["fwd_hr"]["src"][0]["nuse"]     # roi index of right hemisphere start after left hemisphere

    roi_mask = np.zeros(data["fwd_hr"]["nsource"], dtype=bool)
    roi_mask[roi] = True
    
    # generate background
    std        = compute_std(s_snr)
    background = np.random.normal(scale=std, size=(data["fwd_hr"]["nsource"], int(data["eeg_coreg"].info["sfreq"])))      # make it brown signal

    # generate ground truth
    brain_signal      = background
    brain_signal[roi] = signal
    brain_signal      = mne.SourceEstimate(brain_signal, vertno, tmin=0, tstep=1/sfreq, subject=sub)

    # project to eeg
    eeg = mne.apply_forward(data["fwd_hr"], brain_signal, data["eeg_coreg"].info, verbose="warning")
    # ADD EEG NOISE HERE !!!!!
    
    # save simulation
    trial_folder = folder/f"trial{i}"
    trial_folder.mkdir()

    brain_signal.save(trial_folder/f"{sub}_trial{i}_stc.fif", verbose="warning")
    eeg.save(trial_folder/f"{sub}_trial{i}_ave.fif", verbose="warning")

# generate info file
text = f"""
    Easy Simulation \n
    subject: {sub} \n
    duration: {duration} \n
    range for number of dipoles: {rg_num_dipoles} \n
    range for source snr: {rg_s_snr}\n
"""

with open("info.txt", "w") as file:
    file.write(text)

# %%
