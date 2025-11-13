# Cross subject experiments

## Models and combinations

We will evaluate 5 different schemas for the classification, with the idea that CCA can act as a trasnformation that can transform eeg data into a more descriptive feature space:

- CCA: original CCA with classification based solely on highest coefficient
- EEGNETv4: braindecode implementation of EEGNET suited for ssvep
- DNN: Reimplementation of deep neural net for ssvep classification (original code was in matlab and no implementation was found in python)
- CCA as feature extractor:
    - CCA+EEGNET
    - CCA+DNN

## Experiments to be made:

We sill use the benchmark dataset that is comomon on this field, we will test for accuracy and ITR (information transfer rate).

Each subject performed 6 trials for each of the 40 frequencies present on the dataset.

### Leave One Subject Out

In this setup data from one user is kept as test data while all other users data is used to train the models e.g., for 10 users, 1 is used for testing and 9 for training. We separate 10% of the training data to validate and check training metrics such as validation loss and accuracy to get insights into the training, check for overfitting and see how it progesses.

Each trial started with a visual cue for 0.5s folowed by 5 seconds of stumilations recorded and another 0.5s at the end. This totaled 6s for each trial. the visual latency estimated by the authors for this dataset is 140ms, and all captures used a sampling frequency of 1000Hz that was then downsampled to 250Hz to reduce the number of data points.

To get each signal we desconsidered the first 0.5+0.14 seconds of each trial, this corresponds to 160 data points and also disconsidered the last 0.5s (125 samples). For each signal duration we get the corresponding number of time points beginning at the sample 161.

We could increase the number of signals with windowing, but we choose to do not as to compare with results from the original papers and to avoid problems that could arise from overlapping. This could be a later study.

#### Experiments

Frequencies: 8 and 40
Users: 10 and 35
Signal duration: 0.4, 0.6, 0.8, 1s

- [ ] 8 frequencies, 10 users
    - [x] CCA
    - [x] EEGNET
    - [x] DNN
    - [x] CCA+EEGNET
    - [x] CCA+DNN

