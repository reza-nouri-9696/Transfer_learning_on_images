# Transfer_learning_on_images
Here is a README.md for your image classification code and config:

# Image Classification with TensorFlow Hub

This project finetunes a TensorFlow Hub image classification model on a custom dataset.

## Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Training](#training)
- [Evaluation](#evaluation)
- [Models](#models)
- [Contributing](#contributing)

## About <a name = "about"></a>

The code in `finetune_classification.py` trains an image classifier using transfer learning with a TensorFlow Hub model. It supports many different models like EfficientNet, ResNet, Inception, etc. The `config.py` file contains mappings for the model URLs and input image sizes.

The script loads a dataset, trains a model for a given number of epochs, evaluates on a validation set, and saves the fine-tuned model. It also generates plots for training/validation loss and accuracy.

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.6+ and the following libraries:

- matplotlib
- numpy
- tensorflow
- tensorflow_hub
- sklearn

It is recommended to use a [virtualenv](https://docs.python.org/3/library/venv.html) for isolating dependencies.

### Installing

Clone the repository:

```
git clone https://github.com/<username>/image-classification.git
```

## Training <a name="training"></a>

To train a model, run `finetune_classification.py` with the following arguments:

```
python finetune_classification.py --data_dir DATASET --model_name MODEL --epochs N --output_dir OUTPUT
```

- `DATASET`: Path to the training data 
- `MODEL`: Name of the TF Hub model to use (see [Models](#models) below)
- `N`: Number of epochs to train for
- `OUTPUT`: Directory to save plots and trained model

Example:

```
python finetune_classification.py --data_dir data/flowers --model_name efficientnet_b0 --epochs 5 --output_dir training_output
```
###Models
The following TensorFlow Hub models are supported:

efficientnetv2-s
efficientnetv2-m
efficientnetv2-l
efficientnetv2-s-21k
efficientnetv2-m-21k
efficientnetv2-l-21k
efficientnetv2-xl-21k
efficientnetv2-b0-21k
efficientnetv2-b1-21k
efficientnetv2-b2-21k
efficientnetv2-b3-21k
efficientnetv2-s-21k-ft1k
efficientnetv2-m-21k-ft1k
efficientnetv2-l-21k-ft1k
efficientnetv2-xl-21k-ft1k
efficientnetv2-b0-21k-ft1k
efficientnetv2-b1-21k-ft1k
efficientnetv2-b2-21k-ft1k
efficientnetv2-b3-21k-ft1k
efficientnetv2-b0
efficientnetv2-b1
efficientnetv2-b2
efficientnetv2-b3
efficientnet_b0
efficientnet_b1
efficientnet_b2
efficientnet_b3
efficientnet_b4
efficientnet_b5
efficientnet_b6
efficientnet_b7
bit_s-r50x1
inception_v3
inception_resnet_v2
resnet_v1_50
resnet_v1_101
resnet_v1_152
resnet_v2_50
resnet_v2_101
resnet_v2_152
nasnet_large
nasnet_mobile
pnasnet_large
mobilenet_v2_100_224
mobilenet_v2_130_224
mobilenet_v2_140_224
mobilenet_v3_small_100_224
mobilenet_v3_small_075_224
mobilenet_v3_large_100_224
mobilenet_v3_large_075_224
The model_handle_map in config.py contains the TensorFlow Hub URLs for each model.
## Evaluation <a name="evaluation"></a>

The script will evaluate the model on a validation set and report precision, recall, F1 score. It also generates plots for training/validation loss and accuracy.

## Models <a name="models"></a>

A list of supported TF Hub models is defined in `config.py`. This includes EfficientNet, ResNet, Inception, MobileNet, etc.

To use a different model, add it to the `model_handle_map` and `model_image_size_map` dictionaries in `config.py`.

## Contributing <a name="contributing"></a> 

Contributions are welcome! Please open an issue or PR if you would like to add new features or bug fixes.

### TODO

Some ideas for improvements:

- Support more models
- Add options for different optimizers, losses etc
- Logging and TensorBoard support
- Multi-GPU training
- ONNX export
