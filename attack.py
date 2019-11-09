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
import random
from scipy.signal import correlate
def crop_and_resize(arr, x_crop, y_crop, resample, *args, **kwargs):
    """Crop a 2darray and resize the data"""
    
    len_x_crop = x_crop[1]-x_crop[0]
    len_y_crop = y_crop[1]-y_crop[0]

    arr_crop = arr[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
    f = interpolate.interp2d(np.arange(len_y_crop), 
                             np.arange(len_x_crop), 
                             arr_crop, 
                             *args, **kwargs)
    result = f(np.arange(len_x_crop, step=len_x_crop/resample[1]), 
             np.arange(len_y_crop, step=len_y_crop/resample[0]))
    return result
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.01
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i , int(num_salt)) for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i , int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss*0.2
        return noisy
def attack(wmi,wm):
    wm = wm.reshape(256,256)
    # Gaussian filter
    wmi_gauss = gaussian(wmi[0], sigma=1, multichannel= 3)
    wmi_gauss = np.expand_dims(wmi_gauss, axis=0)
    wm_gauss = gaussian(wm ,sigma=1).reshape(1,256,256,1)
    # Rotation by 90 degree
    wmi_rot90 = np.rot90(wmi[0])
    wmi_rot90 = np.expand_dims(wmi_rot90, axis=0)
    wm_rot90 = np.rot90(wm).reshape(1,256,256,1)
    # Rotation by 10 degree
    wmi_rot10 = rotate(wmi,10,axes=(1,2),reshape=False)
    wm_rot10 = rotate(wm,10,axes=(0,1),reshape=False)
    wm_rot10 = wm_rot10.reshape(1,256,256,1)
    # Cropping
    wmi_crop = []
    for i in range(3):
        wmi_crop_ = crop_and_resize(wmi[0,:,:,i].T,(10,118),(10,118),(128,128))
        wmi_crop.append(wmi_crop_)
    wmi_crop= np.array(wmi_crop).T
    wmi_crop = np.expand_dims(wmi_crop, axis=0)
    wm_crop = crop_and_resize(wm,(20,236),(20,236),(256,256))
    wm_crop = wm_crop.reshape(1,256,256,1)
    # Noise
    wmi_sp = noisy('s&p', wmi[0])
    wmi_sp = np.expand_dims(wmi_sp, axis=0)
    wmi_spkl = noisy('speckle',wmi[0])
    wmi_spkl = np.expand_dims(wmi_spkl, axis=0)
    wmi_gnoise = noisy('gauss',wmi[0]) 
    wmi_gnoise = np.expand_dims(wmi_gnoise, axis=0)
    wm = wm.reshape(256,256,1)
    wm_sp = noisy('s&p', wm).reshape(1,256,256,1)
    wm_spkl=noisy('speckle',wm).reshape(1,256,256,1)
    wm_gnoise = noisy('gauss',wm).reshape(1,256,256,1)
    X = np.concatenate((wmi_gauss,wmi_rot90,wmi_rot10,wmi_crop,wmi_sp,wmi_spkl,wmi_gnoise),axis=0)
    X1 = np.concatenate((wm_gauss,wm_rot90,wm_rot10,wm_crop,wm_sp,wm_spkl,wm_gnoise),axis=0)
    return X,X1
def check_attack(X1,x1,name):
    fig, ax = plt.subplots(2, 4, figsize=(16,8))
    ax[0,0].imshow(X1[0].reshape(256,256), cmap='gray')
    ax[0,0].axis('off')
    ax[0,0].set_title('Gaussian Filter')
    ax[0,1].imshow(X1[1].reshape(256,256),cmap='gray')
    ax[0,1].axis('off')
    ax[0,1].set_title('Rotation 90 degree')
    ax[0,2].imshow(X1[2].reshape(256,256),cmap='gray')
    ax[0,2].axis('off')
    ax[0,2].set_title('Rotation 10 degree')
    ax[0,3].imshow(X1[6].reshape(256,256),cmap='gray')
    ax[0,3].axis('off')
    ax[0,3].set_title('Gaussian Noise')
    ax[1,0].imshow(X1[3].reshape(256,256), cmap='gray')
    #ax[0,1].set_title()
    ax[1,0].axis('off')
    ax[1,0].set_title('Cropping')
    ax[1,1].imshow(X1[4].reshape(256,256),cmap='gray')
    ax[1,1].axis('off')
    ax[1,1].set_title('Salt and Pepper Noise')
    ax[1,2].imshow(X1[5].reshape(256,256),cmap='gray')
    ax[1,2].axis('off')
    ax[1,2].set_title('Speckle Noise')
    ax[1,3].imshow(x1.reshape(256,256),cmap='gray')
    ax[1,3].axis('off')
    ax[1,3].set_title('Original watermark')
    fig.savefig('./Attack_Results/'+name)
def norm(wmex,wm):
    if np.max(wmex) != 1:
        wmex = wmex/np.max(wmex)
    if np.max(wm) != 1:
        wm = wm/np.max(wm)
    return wmex,wm
def check_results(WMEX,WM):
    for i in range(np.shape(WMEX)[0]):
        if np.max(WMEX[i]) != 1:
            WMEX[i] = WMEX[i]/np.max(WMEX[i])
    for i in range(np.shape(WM)[0]):
        if np.max(WM[i]) != 1:
            WM[i] = WM[i]/np.max(WM[i])
    a = (WMEX-WM)**2
    mse = a.reshape((a.shape[0],-1)).mean(axis=1)
    psnr = 10*np.log10(1/mse)
    return psnr
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
    batch_size = 7    

    # Adversarial loss ground truths
    valid = np.ones((batch_size,))
    fake = np.zeros((batch_size,))


   
    img_path = 'tulips.png'
    img = image.load_img(img_path, target_size=(128,128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = (x-x.min())/(x.max()-x.min())
    

    alexa = np.zeros((256,257))
    alexa[:242,:] = stft_modified_scaled
    x1 = alexa[:,:-1].reshape(256,256,1)
    x1 = np.expand_dims(x1, axis=0)

    lel = np.concatenate((x1,x1,x1,x1,x1,x1,x1),axis=0)
    X, X1 = attack(x,x1)
    Auto = []
    select = []
    epochs = 30000
    start_time = datetime.datetime.now()
    for epoch in range(10000):
        auto_loss = auto.train_on_batch(X1,X1)
        Auto.append(auto_loss)
        elapsed_time = datetime.datetime.now() - start_time
        print ("[Epoch %d/%d]  [Autoencoder loss: %f time: %s" % (epoch, epochs, auto_loss, elapsed_time))
        if (epoch%500 ==0) :
            # Check on extracter loss
            
            
            # serialize weights to HDF5
            auto.save_weights("./Models/Autoencoder/model"+str(epoch)+".h5")
            print("Saved autoencoder model to disk")
            select.append(auto_loss)
            num = random.randint(0,6)
            img = X1[num]
            fig = plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            img1 = auto.predict(img.reshape(1,256,256,1))
#            plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
            plt.imshow(img1.reshape(256,256),cmap='gray')
            plt.axis('off')
            plt.title('Generated Image')
           
            plt.subplot(1,2,2)
#            plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
            plt.imshow(img.reshape(256,256),cmap='gray')
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
    X1_ = get_layer_output([X1])[0]
    x1_ = get_layer_output([x1])[0]
    
    WMEX,_ = attack(gen.predict([x,x1_]),x1) # Attacked watermarked cover image
    X1__ = np.concatenate((x1_,x1_,x1_,x1_,x1_,x1_,x1_),axis=0)
    gen.load_weights('F:\Rohit\Sem8\SoundinImage\Trial9\gen29999.h5')
    ex.load_weights('F:\Rohit\Sem8\SoundinImage\Trial9\Models\ex1000.h5')
    #%%
    select=[]
    Gloss = []
    Eloss = []
    Dloss = []
    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Condition on B and generate a translated version
        fake_img = gen.predict([X,X1_])

        # Train the discriminators (original images = real / generated = Fake)
        d_loss_real = disc.train_on_batch(X, valid)
        d_loss_fake = disc.train_on_batch(fake_img, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -----------------
        #  Train Generator
        # -----------------
        
        # Train the generators
        feature_x = model.predict(X)
#        feat_disc = disc_feat.predict(x1)
        extract_loss = ex.train_on_batch(WMEX,X1__)
#        valid = valid.reshape(1,1)
        g_loss = combined.train_on_batch([WMEX,X1_] , [valid, X,feature_x])
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
        if (epoch%500 ==0) :
            # Check on extracter loss
            select.append(extract_loss)
            num = random.randint(0,6)
            img = X[num]
            x1_check = X1[num]
            x1_bla = get_layer_output([x1_check.reshape(1,256,256,1)])[0]
            # serialize weights to HDF5
            gen.save_weights("./Models/gen"+str(epoch)+".h5")
            print("Saved generator model to disk")
     
            ex.save_weights("./Models/ex"+str(epoch)+".h5")
            print("Saved extractor model to disk")
            
            fig = plt.figure(figsize=(10,10))
            plt.subplot(2,2,1)
            g_img = gen.predict([img.reshape(1,128,128,3),x1_bla])
#            plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
            plt.imshow(g_img[0])
            plt.axis('off')
            plt.title('Generated Image')
            plt.subplot(2,2,2)
#            plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Actual Cover Image')
            
            wm_ = ex.predict(g_img)
            wm = decoder.predict(wm_).reshape((256,256))
            plt.subplot(2,2,3)
            plt.imshow(wm,cmap='gray')
            plt.axis('off')
            plt.title("Extracted watermark")
            plt.subplot(2,2,4)

            plt.imshow(x1_check.reshape(256,256),cmap='gray')
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
    #%%
    return gen,ex,disc,combined
def psnr(wmex,wm):
    if np.max(wmex) != 1:
        wmex = wmex/np.max(wmex)
    if np.max(wm) != 1:
        wm = wm/np.max(wm)
    mse = np.mean((wmex-wm)**2)
    psnr = 10*np.log10(1/mse)
    return psnr
if __name__ == '__main__':
    gen, ex,disc, combined = train(epochs=30001, batch_size=1, sample_interval=200)
    ex.load_weights('../Trial9/Models/ex25000.h5')
    ex.load_weights('./Models/ex29500.h5')
    WMEX = decoder.predict(ex.predict(X))
    check_attack(WMEX,x1,'Extracted Watermarks after attack training')
    check_attack(X1,x1,'Actual Distorted watermarks')
    psnr = check_results(WMEX,lel)
    #%% Combination of attacks 
    # Gaussian filter and crop
    wm = x1.reshape(256,256)
    wmi = gen.predict([x,x1_])
    wmi_crop = []
    for i in range(3):
        wmi_crop_ = crop_and_resize(wmi[0,:,:,i].T,(10,118),(10,118),(128,128))
        wmi_crop.append(wmi_crop_)
    wmi_crop= np.array(wmi_crop).T
    wmi_crop = np.expand_dims(wmi_crop, axis=0)
    wm_crop = crop_and_resize(wm,(20,236),(20,236),(256,256))
    
    wmi_gauss = gaussian(wmi[0], sigma=1, multichannel= 3)
    wmi_gauss = np.expand_dims(wmi_gauss, axis=0)
    wm_gauss = gaussian(wm ,sigma=1)   
    
    wmi_rot90 = np.rot90(wmi_gauss[0])
    wmi_rot90 = np.expand_dims(wmi_rot90, axis=0)
    wm_rot90 = np.rot90(wm_crop)
    # Rotation by 10 degree
    wmi_rot10 = rotate(wmi_gauss,10,axes=(1,2),reshape=False)
    wm_rot10 = rotate(wm_crop,10,axes=(0,1),reshape=False)
    
    extracted =decoder.predict(ex.predict(wmi_rot90))
    mse_comb = np.mean((extracted.reshape(64,64)-wm_rot90)**2)
    psnr_comb = psnr(extracted,x1.reshape(256,256))    
    
    
    fig = plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
#            plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
    plt.imshow(wmi_rot90[0])
    plt.axis('off')
    plt.title('Distorted watermarked image')

    plt.subplot(1,3,2)
#            plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
    plt.imshow(extracted.reshape(256,256),cmap='gray')
    plt.axis('off')
    plt.title('Extracted Watermark with PSNR '+str(round(psnr_comb,2))+'dB')
    
    plt.subplot(1,3,3)
    plt.imshow(wm.reshape(256,256),cmap='gray')
    plt.axis('off')
    plt.title("Actual watermark")
    fig.tight_layout()
    fig.savefig('./Attack_Results/Gauss and Rotation90.png')
    
    
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
    

    
