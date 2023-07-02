
import json
import numpy as np
import os
import scipy.signal as signal
import subprocess
import sys

from itertools import combinations
from scipy.io import wavfile

### Constants
IMPULSE_RESPONSE_FOLDER = 'ir_data'
FJ_FOLDER = 'faust_juce_data'
# Replace with actual JUCE modules' path
JUCE_MODULES = '/Users/Applications/JUCE/modules' 
WRAPPER_VOICES = 1


class Preprocess:

    def __init__(self, input_file):
        self.sample_rate, self.audio_data = wavfile.read(input_file)
        self.audio_data = self.audio_data.astype(np.float32) # conversion to float32 to perform preprocessing such as windowing
        self.num_channels = self.audio_data.shape[1]

        # normalize each channel independently
        max_value = np.max(np.abs(self.audio_data), axis=0)
        if (max_value > 0).all():
            self.audio_data /= max_value

    # Apply Blackmann window to data
    def apply_window(self, data):
        window_length = len(data)
        window = signal.windows.blackman(window_length)[:, None]
        windowed_data = data * window
        return windowed_data
    
    # Scale to -1. 1. (not recommend for state space model inference)
    # normalize each channel independently
    def apply_normalization(self, data):
        normalized_data = np.empty_like(data, dtype=np.float32)
        max_magnitude = np.max(np.abs(data), axis=0)
        normalized_data = data / max_magnitude
        return normalized_data
    
    # Bandpass filters data according with freq_range
    def apply_filter(self, data, lowcut, highcut, order=4):
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
        filtered_data = signal.sosfilt(sos, data, axis=0)
        return filtered_data

    def preprocessor(self, threshold_percent=0, apply_filter=True, freq_range=[20, 200], apply_window=True, apply_normalization=True):
        # Compute the enrgy of the channel data and set threshold as percentage of maximum energy
        energy = np.sqrt(np.mean(self.audio_data**2, axis=0))
        threshold = threshold_percent * energy ### energy calculated independently on each channel so they are all normalised
        # Compute indices and trim channels
        start_index = np.argmax(np.abs(self.audio_data) > threshold, axis=0)
        end_index = self.audio_data.shape[0] - np.argmax(np.abs(self.audio_data[::-1]) > threshold, axis=0)
        indices = np.r_[start_index, end_index]
        trimmed_data = self.audio_data[np.min(indices):np.max(indices) + 1, :]

        preprocessed_data = trimmed_data

        if apply_filter:
            lowcut = freq_range[0]
            highcut = freq_range[1]
            preprocessed_data = self.apply_filter(preprocessed_data, lowcut=lowcut, highcut=highcut)
        
        if apply_window:
            preprocessed_data = self.apply_window(preprocessed_data)

        if apply_normalization:
            preprocessed_data = self.apply_normalization(preprocessed_data)

        return preprocessed_data



class Analyzer:

    def __init__(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data
        self.num_samples = preprocessed_data.shape[0]
        self.num_channels = preprocessed_data.shape[1]

    def estimate_modes_data(self, threshold=0.0, distance=0.0, sample_rate=44100):

        weights = np.sum(self.preprocessed_data, axis=0) / np.sum(self.preprocessed_data)
        avg_data = np.average(self.preprocessed_data, axis=1, weights=weights)

        modal_freqs, decay_times, modes_gains = self.peak_method(avg_data, threshold, distance, sample_rate)
        
        return modal_freqs, decay_times, modes_gains
    

    def peak_method(self, avg_channels, threshold, distance, sample_rate):

        X = np.abs(np.fft.fft(avg_channels))
        X /= np.max(X)
        dt = 1 / sample_rate
        freqs = np.fft.fftfreq(X.size, dt)

        peak_threshold = np.power(10, np.float32(threshold) / 20.0) # from dB to X unit
        peak_distance = int(distance) / (sample_rate / X.size) # from distance in Hz to samples

        filtered_freqs = []
        filtered_t60s = []
        filtered_gains = []
        n_peaks = 0

        indices = np.argsort(freqs)
        peaks, _ = signal.find_peaks(X[indices], threshold=peak_threshold, distance=peak_distance)

        for p in peaks:
            if freqs[indices][p] > 0:
                freq = freqs[indices][p]
                gain = X[indices][p]

                filtered_freqs.append(freq)
                filtered_gains.append(gain)
                n_peaks += 1

        for i in range(0, n_peaks):

            offset = pow(10, -3/20) # -3 dB to X unit
            pid = peaks[len(peaks) - n_peaks + i] 
    
            n = pid
            while X[indices][n] > (X[indices][pid]*offset):
                n -= 1
            a = n

            n = pid
            while X[indices][n] > (X[indices][pid]*offset):
                n += 1
            b = n

            bandwidth = (b - a) / (sample_rate / X.size)
            filtered_t60s.append((6.91 / sample_rate / (1 - np.exp(-np.pi * bandwidth / sample_rate))) * 150.0)

        frequencies = np.array(filtered_freqs)
        decays = np.array(filtered_t60s)
        gains = np.array(filtered_gains)

        return frequencies, decays, gains


class DataWriter:

    def __init__(self, freqs, t60s, gains, filename, info):
        self.freqs = freqs
        self.t60s = t60s
        self.gains = gains
        self.filename = filename
        self.info = info

    def check_and_create(self, folder_name):
        folder_path = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        return folder_path

    def save_data(self):

        path = self.check_and_create(IMPULSE_RESPONSE_FOLDER)

        data = {
            str(self.info): {
                "freqs": self.freqs.tolist(),
                "T60s": self.t60s.tolist(),
                "gains": self.gains.tolist()
            }
        }

        # Save the dictionary as a JSON file
        self.filename += ".json"
        file_path = os.path.join(path, self.filename)
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=3)

    def save_combinations(self, n):
        # Calculate combinations of n elements from frequencies array
        combinations_list = list(combinations(range(len(self.freqs)), n))

        combination_dict = {}
        for i, indices in enumerate(combinations_list):
            idx = np.array(indices)
            freqs_subset = self.freqs[idx]
            t60s_subset = self.t60s[idx]
            gains_subset = self.gains[idx]

            combination_dict[f"comb_{i+1}"] = {
                "freqs": freqs_subset.tolist(),
                "T60s": t60s_subset.tolist(),
                "gains": gains_subset.tolist()
            }
        
        name = self.filename + "_comb" + str(n) + ".json"
        folder_path = os.path.join(os.getcwd(), IMPULSE_RESPONSE_FOLDER)

        file_path = os.path.join(folder_path, name)
        with open(file_path, "w") as json_file:
            json.dump(combination_dict, json_file, indent=3)

    def write_list(self, v_name, e_name, n_modes, file):
        file.write(f"{v_name} = (")
        i = 0
        while i < n_modes:
            file.write(e_name + str(i+1).zfill(2))
            if i+1 < n_modes:
                file.write(",")
            i += 1
        file.write(");\n")


    def data2dsp(self, nvoices, name):

        nmodes = self.freqs.shape[0]
        
        # Sanity check
        try:
            if nmodes < 1:
                raise ValueError('The data you are pointing to does not contain any information to render!')
            
            filename = name + ".dsp"
            file_path = os.path.join(os.getcwd(), os.path.join(FJ_FOLDER, filename))
            file = open(file_path, "w")

            file.write("/// *****----- FAUST -----***** ///\n")
            file.write("/// this file has been generated by ir2device command line tool ///\n\n")
            file.write("declare copyright \"Alessandro Anatrini - HfMT Hamburg\";\n\n")
            file.write("declare options \"[osc:on]\";\n")
            file.write("declare options \"[midi:on]\";\n")
            file.write("import(\"stdfaust.lib\");\n")
            file.write("import(\"engine.lib\");\n\n")

            file.write("\nn_modes = " + str(nmodes) + ";")
            file.write("\nn_voices = " + str(nvoices) + ";\n")

            i = 0
            while i < nmodes:
                idx = str(i+1).zfill(2)
                file.write("\nmode_" + idx + ' = mode_group(vslider("F' + idx + ' [style:knob][unit:Hz]",' + str(np.min(self.freqs)) + ',' + str(np.min(self.freqs)) + ',' + str(np.max(self.freqs)) + ',0.01));')
                file.write("\nt60_" + idx + ' = t60_group(vslider("D' + idx + ' [style:knob][unit:sec]",' + str(np.min(self.t60s)) + ',' + str(np.min(self.t60s)) + ',' + str(np.max(self.t60s)) + ',0.01)) : si.smoo;')
                file.write("\ngain_" + idx + ' = gain_group(vslider("G' + idx + ' [style:knob]",' + str(np.min(self.gains)) + ',0.0,1.0,0.01)) : si.smoo;')
                file.write("\n")
                i += 1

            file.write('\nattack = global_group(vslider("Attack [style:knob][unit:sec]",0.01,0.01,5.0,0.01));')
            file.write('\ngain = global_group(vslider("Gain [style:knob][unit:dB]",0.0,-70.0,12.0,0.1)) : ba.db2linear : si.smoo;')
            file.write('\ntune = global_group(vslider("Detune [style:knob][unit:cents]",0.0,-100.0,100.0,1.0)) * 0.01;')
            file.write('\nmix = global_group(vslider("Dry Wet [style:knob]",0.5,0.0,1.0,0.01));')
            file.write('\ntrig = button("gate");\n\n')

            self.write_list('modes', 'mode_', nmodes, file)
            self.write_list('decays', 't60_', nmodes, file)
            self.write_list('gains', 'gain_', nmodes, file)
            file.write('\nautogain = 1 / sqrt(n_modes) * gain;')
        
            file.write('\nMODAL_SYNTH = model_wrapper(modes, decays, gains, trig, attack, n_modes, tune, n_voices), (autogain <: _,_) : ro.interleave(2,2) : par(i, 2, *) : ef.dryWetMixerConstantPower(mix, re.dattorro_rev_default);')
            file.write('\nprocess = MODAL_SYNTH;')
            file.write('\neffect = co.limiter_1176_R4_stereo;')

            file.close()
            return file_path

        except ValueError as e:
            print(str(e))
            sys.exit(1)
    
    
    def dsp2juce(self, faust_path):

        cmd = f'faust2juce -midi -osc -nvoices {WRAPPER_VOICES} -jucemodulesdir {JUCE_MODULES} {faust_path}'
        subprocess.call(cmd, shell=True)