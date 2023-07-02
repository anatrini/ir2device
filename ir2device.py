############# Command line tool convert a .wav IR into a JUCE plugin project #####################################
############# Alessandro Anatrini 2021 rev. 2023 ############################################################################################

import argparse
import numpy as np
import sys
from utils import *



### Args parser
parser = argparse.ArgumentParser(description="IR2device creates a device as JUCE project, able to reproduce the modal frequencies, decay times and gains as analysed from an impulse response WAV file. The estimated modal frequencies and their associated values are stored as a JSON file.")

    ### Positional args
parser.add_argument('IRfile', action='store', type=str, help="Path to an impulse_resposne.wav file.")
parser.add_argument('file_name', action='store', type=str, help="Set the name of json file containing the modal analysis.") 
parser.add_argument('synth_name', action='store', type=str, help="Set the name of the JUCE project.")

    ### Optional args:
parser.add_argument('-filter', nargs='+', default=[20, 250], help='Set a bandpass filter for frequency selection. Default=20-250')
parser.add_argument('-thresh', nargs='?', default=-30.0, type=float, help="Set the modes threshold in dB for detection, between -infinity and 0. Default=-30.")
parser.add_argument('-dist', nargs='?', default=0.0, type=float, help="Set the minimum distance in Hertz between two consecutive modes. Default=0.")
parser.add_argument('-comb', nargs='?', default=0, type=int, help='Set length of combinations to be calculated. Deafault=2.')
parser.add_argument('-nmodes', default=4, type=int, help="Set the maximum number of modes. Default=4.")
parser.add_argument('-nvoices', default=8, type=int, help="Set the maximum number of voices of the plug-in. Default=8.")

args = parser.parse_args()

IRfile = args.IRfile
filename = args.file_name
synthname = args.synth_name

freq_range = [int(val) for val in args.filter]
threshold = args.thresh
distance = args.dist
len_comb = args.comb
num_modes = args.nmodes
num_voices = args.nvoices



print("Preprocessing impulse response file...")
prep = Preprocess(IRfile)
sample_rate = prep.sample_rate
prep_data = prep.preprocessor(threshold_percent=0.01, freq_range=freq_range)

print("Analyzing audio data...")
modal_data = Analyzer(prep_data)
modal_freqs, modal_t60s, modal_gains = modal_data.estimate_modes_data(threshold=threshold, distance=distance, sample_rate=sample_rate)

indices = np.argsort(modal_gains)[-num_modes:]
try:
    if len(indices) < num_modes:
        raise ValueError("The number of frequencies to select (-nmodes) exceeds the available indices!")
    sel_freqs = modal_freqs[indices]
    sel_t60s = modal_t60s[indices]
    sel_gains = modal_gains[indices]
except ValueError as e:
    print(str(e))
    sys.exit(1)

name = os.path.splitext(os.path.basename(IRfile))[0]
info = name + "_" + str(freq_range) + "_" + str(threshold) + "_" + str(distance)
dict_ = DataWriter(sel_freqs, sel_t60s, sel_gains, filename, info)
modal_dict = dict_.save_data()

try:
    if len_comb == 0:
        pass
    elif len_comb > 1:
        combs_dict = dict_.save_combinations(len_comb)
    else:
        raise ValueError(f"Invalid value for -comb: 1.")
except ValueError as e:
    print(str(e))
print("The analysis has been stored on the disk.")

faust_path = dict_.data2dsp(num_voices, synthname)
dict_.dsp2juce(faust_path)
print(f"A JUCE project has been saved to {faust_path}")