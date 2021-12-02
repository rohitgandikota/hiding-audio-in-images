# Hiding-Audio-in-Images-using-Deep-Generative-Networks
In this work, we propose an end-to-end trainable model of Generative Adversarial Networks (GAN) which is engineered to hide audio data in images. Due to the non-stationary property of audio signals and lack of powerful tools, audio hiding in images was not explored well. We devised a deep generative model that consists of an auto-encoder as generator along with one discriminator that are trained to embed the message while, an exclusive extractor network with an audio discriminator is trained fundamentally to extract the hidden message from the encoded host signal. The encoded image is subjected to few common attacks and it is established that the message signal can not be hindered making the proposed method robust towards blurring, rotation, noise, and cropping. The one remarkable feature of our method is that it can be trained to recover against various attacks and hence can also be used for watermarking.


[**Link to our paper**](https://link.springer.com/chapter/10.1007/978-3-030-34872-4_43)

To cite our work, please use the following code

@inproceedings{gandikota2019hiding,
  title={Hiding Audio in Images: A Deep Learning Approach},
  author={Gandikota, Rohit and Mishra, Deepak},
  booktitle={International Conference on Pattern Recognition and Machine Intelligence},
  pages={389--399},
  year={2019},
  organization={Springer}
}
