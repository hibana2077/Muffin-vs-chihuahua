# Muffin vs chihuahua

<p align="center">
    <img src="https://skillicons.dev/icons?i=pytorch,py" /><br>
</p>

## Introduction

This is a simple project to predict if an image is a muffin or a chihuahua. The project use vit_small_resnet26d_224 model from timm library.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the following command to train the model:

```bash
cd train
python train.py
```

And you can check the result in the `img` folder.

## Results

The model has an accuracy of 0.98.

![loss](img/loss.png)

## Prediction

![random](./img/predict.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.