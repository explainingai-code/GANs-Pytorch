GAN Implementation in PyTorch
========

This repository implements training and inference methods of GAN with just fc layers on MNIST.

[Paper](https://arxiv.org/pdf/1406.2661.pdf) </br>
[Video on GAN]()
Output on mnist

## Data preparation
For setting up the mnist dataset:

Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

The directory structure should look like this
```
$REPO_ROOT
    -> data
        -> train
            -> images
                -> 0
                    *.png
                -> 1
                ...
                -> 9
                    *.png
        -> test
            -> images
                -> 0
                    *.png
                ...
    -> dataset
    -> tools
        
```

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/GANs-Pytorch.git```
* ```cd GANs-Pytorch```
* ```pip install -r requirements.txt```
* ```python -m tools.train_gan``` for training and saving inference samples

# Training on colored mnist images
* Ensure dataset is prepared according to the Data Preparation instructions

* Change the IM_CHANNELS field to 3 in `train_gan.py`
* Uncomment lines 56-59 in the `dataset/mnist_dataset.py` file

# Training on custom dataset images
* Dump all *.png files(or whatever format images you have) in the path `data/train/images` 
* Directory structure should be following: 
```
data
    -> train
        -> images 
            *.png           
 ```
* Change the `IM_PATH` field to `data/train` in `train_gan.py`
* Change the channels and image sizes accordingly


## Output 
Outputs will be saved every 50 steps in `samples` directory .


During training of GAN the following output will be saved 
* Latest Model checkpoints for generator and discriminator  in ```$REPO_ROOT``` directory

During inference every 50 steps the following output will be saved
* Sampled image grid for in ```samples/*.png``` 






