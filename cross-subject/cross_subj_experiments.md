# Cross subject experiments

## Models and combinations

We will evaluate 5 different schemas for the classification, with the idea that CCA can act as a transformation that can transform eeg data into a more descriptive feature space:

- CCA: original CCA with classification based solely on highest coefficient
- EEGNET: braindecode implementation of EEGNET suited for ssvep
- DNN: Reimplementation of deep neural net for ssvep classification (original code was in matlab and no implementation was found in python)
- CCA as feature extractor:
    - CCA+EEGNET
    - CCA+DNN

## Experiments to be made:

We will use the benchmark dataset that is common on this field, we will test for accuracy and ITR (information transfer rate).

Each subject performed 6 trials for each of the 40 frequencies present on the dataset.

### Leave One Subject Out

In this setup data from one user is kept as test data while all other users data is used to train the models e.g., for 40 users, 1 is used for testing and 39 for training. We separate 15% of the training data to validate and check training metrics such as validation loss and accuracy to get insights into the training, check for overfitting and see how it progresses.

Each trial started with a visual cue for 0.5s followed by 5 seconds of stimulations recorded and another 0.5s at the end. This totaled 6s for each trial. The visual latency estimated by the authors for this dataset is 140ms, and all captures used a sampling frequency of 1000Hz that was then downsampled to 250Hz to reduce the number of data points.

To get each signal we ignore the first 0.5+0.14 seconds of each trial, this corresponds to 160 data points, and also ignore the last 0.5s (125 samples). For each signal duration we get the corresponding number of time points beginning at the sample 161, e.g. for a 1 second window we get 250 time points (from 161 to 411).

We could increase the number of signals with windowing, but we choose to do not as to compare with results from the original papers and to avoid problems that could arise from overlapping. This could be a later study.

### Experiments

#### Hyperparamter tuning

Proposed Best Hyperparameters Per Model

DNN
learning_rate: 0.0004
batch_size: 64
weight_decay: 0.0005

CCA_DNN
learning_rate: 0.001
batch_size: 128
optimizer: Adam
weight_decay: 0.0001

EEGNet
F1: 32
learning_rate: 0.0015
batch_size: 32
optimizer: Adam
weight_decay: 0.0000015
drop_prob: 0.45

CCA_EEGNet
F1: 64
learning_rate: 0.001
batch_size: 32
weight_decay: 0.000075
drop_prob: 0.2

They were based on 20 runs per user with 25 epochs per run.
The parameters were chosen based on the highest score users and those that appeared more times. They were also rounded.

#### Trainings

Frequencies: 40
Users: 35
Signal duration: 0.4, 0.6, 0.8, 1s

- [] 40 frequencies, 10 users
    - [] CCA
    - [] EEGNET
    - [x] DNN
    - [] CCA+EEGNET
    - [] FBCCA+DNN
    - [] custom

