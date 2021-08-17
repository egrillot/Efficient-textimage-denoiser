# Efficient textimage denoiser

It is a set of tools to provide solutions facing noisy and low quality images. You can find 3 packages in this repository using machine learning methods. 

The first **HMM** provides an image segmentation-based method for denoising an image using [hmmlearn](https://github.com/hmmlearn/hmmlearn) library and using methods from the following [repository](https://github.com/jakubcerveny/gilbert) to build a Hilbert-Peano path through the image. This method is the least accurate but the user does not need a data set to denoise his image with it. 

The second package : **CNN** allows the processing of noisy images that have undergone jpeg compression. In this directory, you can find a way to build your own dataset for training the CNN model and an example of use on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. 

Finally, **CRNN** provides a way to generate and manage a dataset of noisy and poor quality text images to train a text recognition model (a CRNN). I generated a database and trained a CRNN on it.

A README is available for each packages. I wish you a good reading ! :)



