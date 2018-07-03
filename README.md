## Data generation using random noise

### Introduction
This project is a part of our course work (EE769). This project aims at expanding any current dataset by generating similar images using GANs concept. This expanded dataset will contribute to other machine learning problems. We take random noise as an input and transform into usable handwriting texts. 

### Results
<table align='center'>
<tr align='center'>
<td> MNIST</td>
<td> EMNIST</td>
</tr>
<tr>
<td><img src = 'mnist/generated images.gif'>
<td><img src = 'emnist/generated_images_EMNIST.gif'>
</tr>
</table>

### Development Environment
* Ubuntu 16.04 LTS
* NVIDIA 840M
* cuda 9.0
* Python 2.7.6
* pytorch 0.4.0
* torchvision 0.1.4
* matplotlib 1.3.1
* imageio 2.2.0
* scipy 0.19.1

### References
Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
(http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

Our code is inspired by:
https://github.com/znxlwm/pytorch-generative-model-collections

