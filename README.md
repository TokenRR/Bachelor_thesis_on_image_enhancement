# Telegram bot for image enhancement
This paper uses:
- Convolutional Neural Network (CNN)
- Residual Dense Block (RDB)
- Telegram bot as a user interface
- ReLU activation function
- [ImageNet](https://www.image-net.org/download.php) dataset taken from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
- Image normalization
- Convert images from RDB to YUV
- Numerical metrics: PSNR, SSIM, MSE, MAE
- Visual comparison of results

## Cloning a repository

To download this code from GitHub, open a terminal and run the following commands:
```sh
git clone https://github.com/TokenRR/Bachelor_thesis_on_image_enhancement.git
cd Bachelor_thesis_on_image_enhancement
```

## Installing dependencies

Create a virtual environment and install dependencies:
```sh
python -m venv venv
source venv/bin/activate  # For Windows, use venv\Scripts\activate
pip install -r requirements.txt
```

## Setting up the configuration
Open the `code/bot/config.py` file and change the bot token to your own:
```python
TOKEN='your-token-here'
```

## Start Telegram bot
To start the Telegram bot, go to the `code/bot` directory and run the following command:
```sh
python main.py
```

## Training models
There are already trained models in the `code\models` directory, but if you want to make changes, you can edit the code in the corresponding Jupyter Notebooks files.