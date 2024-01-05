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
