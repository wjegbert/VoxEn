# VoxEn
A Video Compression Autoencoder using Volumetric Convolutional Kernals

###Justification and Scope
The purpose of this project is to develop a working, *novel* (circa 2018) video compression system using machine learning. While there were a number of research groups working on video compression focused on a combination of recurrent networks or a combination of recurrent and convolutional nodes, this project relies exclusively on convolution to compress video by converting video files into a "cube" of pixels then using 3d convolutional kernals, effectively treating the temporal information as another spatial dimension. Compression was achieved in the encoder network by setting a longer stride across this third quasispatial dimension, allowing the autoencoder to learn to eliminate redundant inter-frame data. The decoder network performs deconvolution (convolutional transpose) using the inverse of the dimensions and strides used for the compressive stage.
This project, in its current form, was completed in 2018 using the old, horrible version of Tensorflow 1, in part because that was the style which had the most documentation available at the time, and in part because I hate myself. This was a project I had no qualifications for and no business working on, and I had no idea what I was doing most of the time. But ain't that always the way. 

###Results and Excuses
The autoencoder was trained on a portion of the HMDB51 video dataset. This dataset consists of thousands of low-res videos of people moving around and doing fun stuff engineers don't have time to do. The specific subset of this dataset was people performing handstands.
Some of our results were pretty good:

| Ground Truth Video | Reconstructed Video Clip |
|------|------|
| ![alt text](GIFs/handstand_groundtruth.npy.gif) | ![alt text](GIFs/31668_reconstructed_loss_192.44221_reluout.npy.gif) |




I would have continued to refine the project, but I was using a lab desktop where I was stuck in a cycle of obsessively tweaking hyperparameters and watching for "number go down," so they changed to password on the Linux partition I was using.
