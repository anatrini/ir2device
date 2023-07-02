# ir2device

 Impulse response to JUCE project command-line tool.


## Introduction

ir2deivice allows to estimate the modal frequencies, decay rates and the gains of resonant modes by analysing an impulse response recording in `wav` format. The analysis is stored on a disk and then used to create a `dsp` file (Faust). This is then converted to a JUCE project that can be compiled by the user as AU or VST plug-in using for example Xcode or VS Code.


## Requirements

In order to use ir2device the following additional pieces of software are required:

- Faust (Functional Audio Stream) available at https://github.com/grame-cncm/faust 
- JUCE framework available at https://github.com/juce-framework/JUCE

Please follow the instructions on the github repositories to install Faust and JUCE. Concerning Faust ensure that the faust2[...] tools are available.


## Usage

1. In order to use ir2device clone this repository to your machine. It is recommended to run the script inside a virtual environment e.g. Miniconda or Poetry. 
Using conda for example, once you cloned the repository `cd` to the root folder and run:

    `conda create --name <environment_name> python==3.9.16`

    `conda activate <myenv>`

    `pip install -r requirements.txt`

Ensure that `pip` is available within your environment and make sure to activate the environment before installing the requirements.

2. To get an overview of the script's arguments, type in the terminal:

    `python ir2device.py -h`

3. To generate a JUCE project type:

    `python ir2device.py <path_to_impulse_response_file> <analysis_filename> <juce_project_filename>`

The tool will create an "ir_data" folder containing a `json` file with the result of the analysis. If the the `-comb` optional argument is at lest 2 an additional file containing all combinations of lenght equal to `comb` will be created inside the folder. Impulse response files with more than one channel will be automatically downmixed to mono through a weighted average run across all available channels.

4. The `dsp` file and the JUCE project will be saved inside the "faust_juce_data" folder.
