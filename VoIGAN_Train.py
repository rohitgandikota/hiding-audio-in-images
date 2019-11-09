# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:33:19 2019

@author: IIST
"""
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras import layers
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid, tanh
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import cv2
from keras.preprocessing import image
from pretrained import VGG19
from keras.models import model_from_json
import h5py
from skimage import filters
import scipy
from scipy.signal import correlate
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
def build_generator():
    """U-Net Generator"""
    gf = 64 
    channels = 3
    img_shape = (128,128,3)
    watermark_shape = (2,2,1024)
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u
    # Image input
    d0 = Input(shape=img_shape)
    a0 = Input(shape= watermark_shape)
    # Downsampling
    d1 = conv2d(d0, gf, bn=False) #64*64*64
    d2 = conv2d(d1, gf*2) #32*32*128
    d3 = conv2d(d2, gf*4) #16*16*256
    d4 = conv2d(d3, gf*8) #8*8*512
    d5 = conv2d(d4, gf*8) #4*4*512
    d6 = conv2d(d5, gf*16) #2*2*1024
    d6 = Concatenate()([d6, a0])
    #Upsample
    u2 = deconv2d(d6, d5, gf*8)
    u3 = deconv2d(u2, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)
    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh',name='conv6')(u7)
    gen = Model([d0,a0],output_img)   

    return gen
def build_autoencoder():
    """U-Net Generator"""
    gf = 64 
    channels = 1
    img_shape = (256,256,1)
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
   

    def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        return u
    
    # Image input
    d0 = Input(shape=img_shape)
    
    # Downsampling
    d1 = conv2d(d0, gf, bn=False) #128*128*64
    d2 = conv2d(d1, gf*2) #64*64*128
    d3 = conv2d(d2, gf*4) #32*32*256
    d4 = conv2d(d3, gf*8) #16*16*512
    d5 = conv2d(d4, gf*8) #8*8*512
    d6 = conv2d(d5, gf*8) #4*4*512
    d7 = conv2d(d6, gf*16) #2*2*1024
    # Upsampling
#    u1 = deconv2d(d7, d6, gf*8)
    u2 = deconv2d(d7, gf*8)
    u3 = deconv2d(u2, gf*8)
    u4 = deconv2d(u3, gf*8)
    u5 = deconv2d(u4, gf*4)
    u6 = deconv2d(u5, gf*2)
    u7 = deconv2d(u6, gf)
    u8 = UpSampling2D(size=2)(u7)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='relu',name='conv6')(u8)
    autoencoder = Model(d0,output_img)   
    d = Input(shape=(2,2,1024))
    for i in range(20):
        if i == 0:
            a = autoencoder.layers[-(20-i)](d)
        else:
            a = autoencoder.layers[-(20-i)](a)
    decoder = Model(d,a)
    return autoencoder, decoder
def build_extracter():
    """U-Net Generator"""
    gf = 64 
    img_shape = (128,128,3)
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u
    # Image input
    d0 = Input(shape=img_shape)
    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*8)
    d6 = conv2d(d5, gf*16)
    
    ext = Model(d0,d6)
    
    return ext#, emb

def build_discriminator():
    df = 16
    img_shape = (128,128,3)
    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_A = Input(shape=img_shape)

    # Concatenate image and conditioning image by channels to produce input

    d1 = d_layer(img_A, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*4)
    d5 = d_layer(d4, df*8)
    d6 = d_layer(d5, df*8)
    d7 = d_layer(d6, df*16)
    flat = Flatten()(d7) 
    
    validity = Dense(1, activation='sigmoid')(flat)
    disc = Model(img_A,validity)
    return disc
#def build_feat_discriminator():
#    df = 32
#    img_shape = (256,256,1)
#    def d_layer(layer_input, filters, f_size=4, bn=True):
#        """Discriminator layer"""
#        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
#        d = LeakyReLU(alpha=0.2)(d)
#        if bn:
#            d = BatchNormalization(momentum=0.8)(d)
#        return d
#
#    img_A = Input(shape=img_shape)
#
#
#    d1 = d_layer(img_A, df, bn=False)
#    d2 = d_layer(d1, df*2)
#    d3 = d_layer(d2, df*4)
#    d4 = d_layer(d3, df*8)
#    d5 = d_layer(d4, df*8)
#
#    disc = Model(img_A,d5)
#    return disc
#

def train(epochs, batch_size=1, sample_interval=50):
    batch_size = 1
    img_shape = (128,128,3)
    watermark_shape = (2,2,1024)
    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((batch_size,))
    fake = np.zeros((batch_size,))

    optimizer = Adam(0.0002, 0.5)
    optimizer2 = Adam(0.002, 0.5)
    
    # Build and compile the discriminator
    disc = build_discriminator()
    disc.compile(loss='mse',
        optimizer=optimizer,
        metrics=['accuracy'])
#    
#    disc_feat = build_feat_discriminator()
#    disc_feat.compile(loss='mse',
#        optimizer=optimizer,
#        metrics=['accuracy'])    
    #-------------------------
    # Construct Computational
    #   Graph of Generator
    #-------------------------
    
    # Build the generator
    gen = build_generator()
    ex = build_extracter()
    # Input images and their conditioning images
    img_A = Input(shape=img_shape)
    img_B = Input(shape = watermark_shape)
    # By conditioning on B generate a fake version of A
    fake_A = gen([img_A,img_B])
    
    #Generating spectrogram
    
    # For the combined model we will only train the generator
    disc.trainable = False
#    disc_feat.trainable = False
    # Discriminators determines validity of translated images / condition pairs
    valido = disc(fake_A)
#    valido_feat = disc_feat(spect)
    model = VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    feature = model(fake_A)
    ex.compile(loss='mse', optimizer=optimizer2)
    combined = Model(inputs=[img_A,img_B], outputs=[valido, fake_A,feature])
    combined.compile(loss=['mse', 'mae','mse'],
                          loss_weights=[1, 100,100],
                          optimizer=optimizer)
#    combined_feat = Model(inputs=img_A, outputs=[valido_feat])
#    combined_feat.compile(loss=['mse'], optimizer=optimizer)
##    featured.compile(loss='mse', optimizer=optimizer)
   
    img_path = 'tulips.png'
    img = image.load_img(img_path, target_size=(128,128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = (x-x.min())/(x.max()-x.min())
    
#    img_path = 'watermark.jpg'
#    img = image.load_img(img_path, target_size=(64, 64))
#    x1 = image.img_to_array(img)
#    x1 = rgb2gray(x1)
#    x1 = x1.reshape((2,2,1024))
#    x1 = np.expand_dims(x1, axis=0)
#    x1 = (x1-x1.min())/(x1.max()-x1.min())
#    img = x1.reshape(64,64)*255
#    thres = filters.threshold_mean(img)
#    img[img>thres] = 255
#    img[img<thres] = 0    
#    x1 = img.reshape((1,2,2,1024)) / 255
    alexa = np.zeros((256,257))
    alexa[:242,:] = stft_modified_scaled
    x1 = alexa[:,:-1].reshape(256,256,1)
    x1 = np.expand_dims(x1, axis=0)
    select = []
    epochs = 30000
    #build autoencoder to fit data
    auto,decoder = build_autoencoder()
    auto.compile(loss='mse', optimizer=optimizer)   
    # Fit autoencoder
    Auto = []

    for epoch in range(epochs):
        auto_loss = auto.train_on_batch(x1,x1)
        Auto.append(auto_loss)
        elapsed_time = datetime.datetime.now() - start_time
        print ("[Epoch %d/%d]  [Autoencoder loss: %f time: %s" % (epoch, epochs, auto_loss, elapsed_time))
        if (epoch%500 ==0) :
            # Check on extracter loss
            select.append(auto_loss)

            # serialize weights to HDF5
            auto.save_weights("./Models/Autoencoder/model"+str(epoch)+".h5")
            print("Saved autoencoder model to disk")
     
            fig = plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            img = auto.predict(x1)
#            plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
            plt.imshow(img.reshape(256,256),cmap='gray')
            plt.axis('off')
            plt.title('Generated Image')
            c_img = x1
            plt.subplot(1,2,2)
#            plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
            plt.imshow(c_img.reshape(256,256),cmap='gray')
            plt.axis('off')
            plt.title('Actual Image')
            fig.savefig('./Autoencoder/Test_epoch'+str(epoch)+'.png')
    select=[]
    Gloss = []
    Eloss = []
    Dloss = []
#    D_feat = []
    from keras import backend as K
    # with a Sequential model
    get_layer_output = K.function([auto.layers[0].input],
                                      [auto.layers[19].output])
    x1_ = get_layer_output([x1])[0]
    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Condition on B and generate a translated version
        fake_img = gen.predict([x,x1_])

        # Train the discriminators (original images = real / generated = Fake)
        d_loss_real = disc.train_on_batch(x, valid)
        d_loss_fake = disc.train_on_batch(fake_img, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -----------------
        #  Train Generator
        # -----------------

        # Train the generators
        feature_x = model.predict(x)
#        feat_disc = disc_feat.predict(x1)
        extract_loss = ex.train_on_batch(fake_img,x1_)
#        valid = valid.reshape(1,1)
        g_loss = combined.train_on_batch([x,x1_] , [valid, x,feature_x])
#        d_feat_loss =combined_feat.train_on_batch(x,feat_disc)
#        f_loss = featured.train_on_batch(x,feature_x)
        elapsed_time = datetime.datetime.now() - start_time
        Gloss.append(g_loss[0])
        Eloss.append(extract_loss)
        Dloss.append(d_loss[0])
#        D_feat.append(d_feat_loss)
        # Plot the progress
        print ("[Epoch %d/%d]  [D loss: %f, acc: %3d%%]  [G loss: %f] [E loss: %f] time: %s" % (epoch, epochs,
                                                                d_loss[0], 100*d_loss[1],  
                                                                g_loss[0], extract_loss,
                                                                    elapsed_time))
        if (epoch%100 ==0) :
            # Check on extracter loss
            select.append(extract_loss)

            # serialize weights to HDF5
            gen.save_weights("./Models/gen"+str(epoch)+".h5")
            print("Saved generator model to disk")
     
            ex.save_weights("./Models/ex"+str(epoch)+".h5")
            print("Saved extractor model to disk")
            
            fig = plt.figure(figsize=(10,10))
            plt.subplot(2,2,1)
            g_img = gen.predict([x,x1_])
#            plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
            plt.imshow(g_img[0])
            plt.axis('off')
            plt.title('Generated Image')
            c_img = x
            plt.subplot(2,2,2)
#            plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
            plt.imshow(c_img[0])
            plt.axis('off')
            plt.title('Actual Cover Image')
            
            wm_ = ex.predict(x)
            wm = decoder.predict(wm_).reshape((256,256))
            plt.subplot(2,2,3)
            plt.imshow(wm,cmap='gray')
            plt.axis('off')
            plt.title("Extracted watermark")
            plt.subplot(2,2,4)

            plt.imshow(x1.reshape(256,256),cmap='gray')
            plt.axis('off')
            plt.title("Actual Watermark")
            fig.savefig('./Test_epoch'+str(epoch)+'.png')
    model_json = gen.to_json()
    with open("gen"+str(epoch)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    gen.save_weights("gen"+str(epoch)+".h5")
    print("Saved generator model to disk")
    decoder.save_weights("./Models/decoder"+str(epoch)+".h5")
    print("Saved Decoder model to disk")
    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(Gloss)
    
    plt.title('Generator Loss')
    plt.subplot(1,3,2)
#            plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
    plt.plot(Dloss)
    
    plt.title('Discriminator Loss')
    plt.subplot(1,3,3)
    plt.plot(Eloss)
    
    plt.title('Extracter Loss')
#    plt.subplot(1,4,4)
#    plt.plot(D_feat)    
#    plt.title("Extractor Loss")
    fig.savefig('loss_curves.png')
    return gen,ex,disc,combined

if __name__ == '__main__':
    gen, ex,disc, combined = train(epochs=30001, batch_size=1, sample_interval=200)
    ex.load_weights('./Models/ex1000.h5')
    wm = ex.predict(x)
    wm = decoder.predict(wm)
    wm = wm.reshape((256,256))
    alexa = np.zeros((256,257))
    alexa[:,:-1] = wm
    x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(alexa[:242,:],
                                                                   fft_size, hopsamp,
                                                                   iterations)

    # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    max_sample = np.max(abs(x_reconstruct))
    if max_sample > 1.0:
        x_reconstruct = x_reconstruct / max_sample

    # Save the reconstructed signal to a WAV file.
    audio_utilities.save_audio_to_file(x_reconstruct, sample_rate_hz,outfile='Final_reconstruction1000.wav')


    ######################## Do the spec to speech and save it
    
    fs, y = scipy.io.wavfile.read('perhaps.wav')
    fs, y_hat = scipy.io.wavfile.read('Final_reconstruction1000.wav')
    y = y / np.max(abs(y))
    y_hat = y_hat / np.max(abs(y_hat))
    
    fig = plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(y)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
#    plt.axis('off')    
    plt.title('Original Speech signal')
    plt.subplot(1,2,2)
    plt.plot(y_hat)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Extracted Speech signal')
    fig.savefig('Extraction_speech1000.png')
    
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(x1.reshape(256,256),cmap='gray')
    plt.axis('off')    
    plt.title('Original Spectrogram')
    plt.subplot(1,2,2)
    plt.imshow(wm,cmap='gray')
    plt.axis('off')    
    plt.title('Extracted Spectrogram')
    fig.savefig('Extraction_spectrogram1000.png')

    xcorr = correlate(y, y_hat)
    nsamples = np.size(y)
    # delta time array to match xcorr
    dt = np.arange(1-nsamples, nsamples)
    recovered_time_shift = dt[xcorr.argmax()]
    y_hat = np.roll(y_hat,recovered_time_shift)
    mse = np.mean((y-y_hat)**2)
    psnr = 10*np.log10(1/mse)
    cov = np.corrcoef(y,y_hat)
    

    
