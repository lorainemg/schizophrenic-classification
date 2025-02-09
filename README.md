# Schizophrenic Classification

This project focuses on the classification of schizophrenic patients using the wavelet transform applied to evoked potentials. It utilizes the P300 visual paradigm and the Daubechies wavelet transform (order 4, level 5) to extract features. Support Vector Machine (SVM) is used for classification with cross-validation.

## Background

Schizophrenia is a psychiatric disorder that affects a large portion of the world’s population. Among its symptoms is the inability to process information, mainly in attention tasks and work memory. That is why evoked potentials related to events are useful tools for supporting the subjective decision-making processes of medical specialists. The objective of this work is to generate a model that allows distinguishing schizophrenic patients from healthy patients using the visual paradigm of the P300 and applying the discrete transform of wavelets as a method of extracting characteristics.

## Features

- Classification of schizophrenic patients from healthy subjects
- Uses P300 visual paradigm and wavelet transform
- The discrete transform of wavelets as a method of extracting characteristics
- The database used has records of evoked potentials of 54 healthy subjects and 54 patients matched by age and sex
- The discrete wavelet transform was applied with the Daubechies mother wavelet of order 4 and level 5
- 180 characteristics were extracted
- SVM was applied as a learning algorithm using cross-validation to obtain a 62.93% accuracy.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/lorainemg/schizophrenic-classification.git
cd schizophrenic-classification
pip install -r requirements.txt
``` 

## Documentation

A full report about the procedure can be found at [paper.pdf](https://github.com/lorainemg/schizophrenic-classification/blob/main/doc/paper.pdf).
