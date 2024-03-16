# Fakemon

## Introduction

The goal of the project is to make a neural network which would be able to generate images of new pokemons thats doesn't show in original series. The program should be able to create a 2D images of resolution at least 512x512 pixels and the images should remind real pokemons. 

## Architecture

To achieve this goal, we will implement Generative Adversial Network - both Generator and Discriminator. Generator is responsible for generating an image from a random vector 
( noise ) and Discrimanotor tries to predict if gives image is real or fake. The most simple appoach is to create a DCGAN - Deep Convolutional GAN. However it might not give us as good results as we would like to. In current state of art, most of the image-generating apps are created using StyleGAN. On of the newest approach uses StyleSwin as a main network in Generator. It uses Video Transormest instead of CNNs and is still a new technology under develop.

## Datasets

We gathered a 2 datasets and merged them into one, containg around 3500 images. There are only around 1000 species of pokemons, so in our dataset there are multiple images on one Pokemon. Firstly we had to normalize the images, as not all of them were in the same format or shape. Some of them also has differentr backgrounds color - we changed every background into black one, because it's less visible for neural network (the weights for black background are close to 0). After that we extended our dataset using data augmentation methods - images reflection, changing the contrast, adding random noise to the image. Using these method, we gathered around 29_000 which wil be our starting point in training the models. 
