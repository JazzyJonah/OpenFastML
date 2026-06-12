# OpenEGammaCNN

Fast machine learning with QKeras for electron and photon classification using open calorimeter data. This project trains a quantised convolutional neural network to classify electron and photon signal objects against jet background objects using small calorimeter-tower images. The model is designed to be lightweight and suitable for fast inference studies.

Run source setup.sh. This will install a light-weight version of conda and along with all dependencies conda activate fastml4jets sets up the environment for this repository.

This repository uses the fastml package for fast tower building and CNN layers (https://cds.cern.ch/record/2941096)

## Overview

The input data consists of calorimeter tower images constructed from simulated events with pile-up corresponding to a 200 pile-up scenario. Each seed is represented as a small calorimeter image centred on a selected tower. The model uses these images to distinguish signal-like electron and photon objects from background-like jet activity.

The final classifier outputs a value between 0 and 1:

* `1`: signal-like seed
* `0`: background-like seed

## Data Format

The dataset is built from calorimeter towers. Each tower corresponds to a coarse region of 0.1 x 0.1 in eta-phi. The towers span the detector region: |eta| < 2.5, |phi| < pi.

Each tower contains energy deposits in two calorimeter layers:

* Electromagnetic layer, `EM`
* Hadronic layer, `HAD`

The CNN input is a local image made from 3 x 3 x 2 calorimeter towers where:

* `3` towers are used in eta
* `3` towers are used in phi
* `2` calorimeter layers are used: EM and HAD

Therefore, each input seed has shape (3, 3, 2).

## Seed Definition

A seed is a local calorimeter image centred on a tower that satisfies the signal or background selection criteria.

A four-momentum-like coordinate convention is used for each seed:

```text
ET  = central tower transverse energy in the EM layer
eta = central tower eta
phi = central tower phi
m   = 0
```

Signal seeds are built from the simulated `Z -> ee` sample. For the signal sample, the central tower is selected from towers that have non-zero energy in the electromagnetic layer in the corresponding zero pile-up sample.

Background seeds are built from the simulated JZ sample. For the background sample, the central tower must satisfy EM ET > 10 GeV in the 200 pile-up sample.

For both signal and background samples, a mask is applied to ensure that selected `3 x 3 x 2` seed images do not overlap.

The resulting non-overlapping images are referred to as seeds.

## CNN Architecture

The CNN performs symmetric depthwise convolutions on the input images. The input shape is (3, 3, 2). The first layer performs symmetric pooling using non-trainable convolutions. This layer applies three symmetric `3 x 3` convolution patterns in the eta-phi plane. The output shape is: (1, 1, 2, 3) where:

* `2` corresponds to the two calorimeter layers
* `3` corresponds to the three symmetric pooling outputs

The pooled outputs are passed to a dense layer. For each calorimeter layer, a fully connected matrix of shape (3, y) is applied, where y = depth multiplicity. The depth multiplicity is a tunable hyperparameter. The output shape after this stage is (1, 1, 2 * y). The output is then passed through a ReLU activation function. The ReLU output is passed to a final dense layer with shape (24, 1). The final output is passed through a sigmoid activation function, giving a classifier score in the range [0, 1].

All trainable layers are quantised using QKeras. The convolution and dense layer precisions are treated as hyperparameters:

```text
conv_precision
dense_precision
```

The ReLU precision is set to match the convolution precision. The bit precision is defined as:

```text
bits = precision_hyperparameter
integer = precision_hyperparameter // 2
```
## Training Dataset Construction

Training uses:

```text
20,000 signal events
10,000 background events
```

From these events, `3 x 3 x 2` seed images are constructed. Signal seeds from the `Z -> ee` sample are labelled as `1`. Background seeds from the JZ sample are labelled as `0`.

The dataset is balanced in seed transverse energy. For each seed `ET` bin between 10 GeV < ET < 40 GeV, the number of signal and background seeds is matched. This prevents the model from learning only the difference in the seed `ET` distributions between signal and background.

After balancing, each seed is assigned a weight using an exponential spline function. The purpose of this weighting is to prioritise lower-`ET` seeds during training, since these seeds are more important for low-threshold trigger performance.

The selected seeds are split into training and validation samples using an 80-20 split. The training data is saved as train_data.parquet with the following fields:

```text
x_train : seed images with shape (3, 3, 2)
y_train : binary labels, 0 or 1
w_train : seed weights
```

The validation data follows the same convention:

```text
x_val : validation seed images
y_val : validation labels
w_val : validation weights
```

Signal and background seeds in the training sample are augmented by a factor of 4. This is done by mirroring the seed images in eta and phi. The corresponding labels and weights are augmented in the same way. The validation sample is not augmented. Both the training and validation samples are batched with batch_size = 512

The baseline model is trained using the following configuration:

```json
{
  "depth_mult": 4,
  "conv_precision": 12,
  "dense_precision": 12,
  "learning_rate": 5e-4
}
```

The training setup is:

```text
epochs = 45
loss = Binary Cross Entropy
optimizer = Adam
```


Hyperparameter tuning is performed over the model hyperparameters. The tunable parameters include:

```text
depth_mult
conv_precision
dense_precision
learning_rate
```

The tuner trains each trial using the same training setup as the baseline model:

```text
epochs = 45
loss = Binary Cross Entropy
optimizer = Adam
```

The number of random hyperparameter trials is specified by the user. The baseline and the best tuned model are saved in `.keras` format.

## Testing Data

The test dataset is stored in two separate Parquet files: test_signal.parquet and test_background.parquet. Each file is saved as an event-structured Awkward Array. The structure is:

```text
N events -> y seeds per event
```

where:

* `N` is the number of events
* `y` is the variable number of selected seeds in each event

Each seed contains two fields: image and seed_info. The `image` field contains the calorimeter seed image associated with each selected seed. The `seed_info` field contains the four-momentum information of the seed. Therefore, each entry in the test dataset links one seed image to its corresponding seed-level kinematic information.