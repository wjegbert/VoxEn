# VoxEn
A Video Compression Autoencoder using Volumetric Convolutional Kernals

## Justification and Scope
The purpose of this project is to develop a working, *novel* (circa 2018) video compression system using machine learning. While there were a number of research groups working on video compression focused on a combination of recurrent networks or a combination of recurrent and convolutional nodes, this project relies exclusively on convolution to compress video by converting video files into a "cube" of pixels then using 3d convolutional kernals, effectively treating the temporal information as another spatial dimension. Compression was achieved in the encoder network by setting a longer stride across this third quasispatial dimension, allowing the autoencoder to learn to eliminate redundant inter-frame data. The decoder network performs deconvolution (convolutional transpose) using the inverse of the dimensions and strides used for the compressive stage.
This project, in its current form, was completed in 2018 using an old, horrible version of Tensorflow 1, in part because that was the style which had the most documentation available at the time, and in part because I hate myself. This was a project I had no qualifications for and no business working on, and I had no idea what I was doing most of the time. But ain't that always the way. 

## Results and Excuses
The autoencoder was trained on a portion of the HMDB51 video dataset. This dataset consists of thousands of low-res videos of people moving around and doing fun stuff engineers don't have time to do. The videos were truncated to 64 frames of 240x320 because anything more would cause the lab desktop we were using to burst into flames.

Some of our results were pretty decent:

| Ground Truth Video | Reconstructed Video Clip |
|------|------|
| ![alt text](GIFs/handstand_groundtruth.npy.gif) | ![alt text](GIFs/handstand_reconstructed_loss_207.78262.npy.gif) |

Some less so:

| Ground Truth Video | Reconstructed Video Clip |
|------|------|
| ![alt text](GIFs/pizzatime_groundtruth.npy.gif) | ![alt text](GIFs/pizzatime_reconstructed_loss_881.7008.npy.gif) |

One thing I noticed is that video clips with less overall optical flow had better reconstruction fidelity than clips with more moving pixels, which is true for traditional video codecs as well. You can see this above, as the pizza clip has a moving camera whereas the handstand clip has a static background.
The trained network was surprising well generalized despite being trained on only a subset of the video files, specifically the subfile for handstands. Feeding the network videos of unrelated motions resulted in surpising un-handstand-y results:
| Ground Truth Video | Reconstructed Video Clip |
|------|------|
| ![alt text](GIFs/archer_groundtruth.npy.gif) | ![alt text](GIFs/archer_reconstructed_loss_251.27802.npy.gif) |

Of course the archer reconstruction was decent because the ground truth is basically a static image, but it demonstrates the generalization nonetheless. I had believed that the network would overfit to the handstand dataset, but it appears that the information in those videos was broad enough to allow it to compress any video, so long as they were 64 frames
I should also note that, after making several examples, I noticed that the coloration on many was all wrong. This was because OpenCV, which I used for the pre and post processing, encodes color data using BGR format, not RGB like the rest of the world. I do not believe this effected the autoencoder training since the meaning of color channels is irrelevant to the network, but I still think it is worth noting since not enough people talk about why OpenCV would do such a cruel prank on people who are just trying to process videos.

The professor who recruited me to this project ultimately found these results disappointing, since the reconstructed videos look like they were filmed underwater in a polluted river by someone with shaky hands. I was an undergrad with a single MNIST tutorial under my belt before I took this project so idk what you expect, but consider this: this thing, this unliving, mindless thing *learned.* It learned rebuild video, to shape pure noise---the inchoate chaos that underpins our reality---into an intelligible video of a person doing a handstand. Is that not how legends of old say deities made the world? Does that not verge on the miraculous? Does that not deserve praise? Does that not deserve at least an A-? 

## Closing Remarks and "Future Work" (lol)
There are a number of possible avenues which could improve performance of this architecture. The most obvious would be to employ background subtraction and focus only on compressing the moving pixels in the video, but increasing the sharpness of the training data and using higher quality or uncompressed videos would likely also improve results. Really any pre or post processing would improve things.
I would have continued to refine the project, but I became stuck in a cycle of obsessively tweaking hyperparameters and watching "number go down," so my friends, in an effort to get me to shower, changed to password on the Linux partition I was using. I probably could have gotten further along in the project if my partner had actually done his part and wrote the code to convert the videos into numpy arrays when I told him to instead of waiting until the last minute. That's right Nick. You had one job and you waited till the last minute, and now you're being called out on Github. I will never not be salty about that.
