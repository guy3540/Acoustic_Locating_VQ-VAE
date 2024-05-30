# Single Source Acoustic Location, Using VQ-VAE

## Overview
This repository contains the implementation of our project, aimed at estimating the location of a speaker in a room using a single microphone. The approach leverages two Vector Quantized Variational Autoencoders (VQ-VAEs) to learn speech features and room impulse response (RIR) features. The project demonstrated promising results, as is elaborated in the project's report.

## Repository Structure
The repository is structured as follows:

* models: Contains the trained VQ-VAE models.
* src: Contains data preprocessing and visualization functions.
  * rir_dataset_generator: Contains the dataset class for generating RIR datasets.
  * vq_vae: Contains the classes used to create our models, including the encoder, decoder, vector quantizer, etc.
* scripts: Contains the training functions and the dataset generation script.
* scratch_scripts: a sandbox for scripts that are supposed to do a specific demonstration, such as creating a sound file from a spectrogram.

## Usage
### Data Preprocessing and Visualization
Scripts for data preprocessing and visualization are located in the src folder.

### Dataset Generation
To generate the RIR dataset, navigate to the scripts folder and run the dataset generation script:
```
python scripts/generate_rir_dataset.py
```
Make sure to update the desired parameters, such as dataset size

### Training the Models
Training functions are available in the scripts folder. For example, to train the speech model, run:
```
python scripts/train_speech.py
```
To make changes in the model architecture, make sure to update the desired parameters in the file itself.

For further questions, please contact:

Guy Shkury: [[guy3540@gmail.com](mailto:guy3540@gmail.com)]

Reie Matza: [[reiematza@gmail.com](mailto:reiematza@gmail.com)]

