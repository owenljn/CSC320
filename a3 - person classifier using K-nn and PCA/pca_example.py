
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os

#os.chdir('c:/Users/Guerzhoy/Desktop/csc320/pca_example/')

#%matplotlib



def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
        From: Jan Erik Solem, Programming Computer Vision with Python
        #http://programmingcomputervision.com/
    """
    
    # get dimensions
    num_data,dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_X



    

def get_digit_matrix(img_dir):
    im_files = [img_dir + filename for filename in os.listdir(img_dir) if filename[-4:] == ".jpg"]
    im_shape = array(imread(im_files[0])).shape[:2] # open one image to get the size 
    im_matrix = array([imread(im_file).flatten() for im_file in im_files])
    im_matrix = array([im_matrix[i,:]/(norm(im_matrix[i,:])+0.0001) for i in range(im_matrix.shape[0])])
    return (im_matrix, im_shape)
    

def get_reconstruction(V, im, mean_im):
    coefs = [np.dot(V[i,:], (im-mean_im)) for i in range(V.shape[0])]

    
    new_im = mean_im.copy()
    for i in range(len(coefs)):
        new_im = new_im + coefs[i]*V[i, :]
    return new_im



def display_save_25_comps(V, im_shape):
    '''Display 25 components in V'''
    figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        gray()
        imshow(V[i,:].reshape(im_shape))
        
        

        
        
def salt_and_pepper_noise(flattened_im, noise_prop):
    im = flattened_im.copy()
    pix_inds = range(len(im))
    perm_inds = np.random.permutation(pix_inds)
    
    im[perm_inds[:int(0.5*noise_prop*len(im))]] = max(im)
    im[perm_inds[int(0.5*noise_prop*len(im)):int(noise_prop*len(im))]] = min(im)
    
    return array(im)


        
def occlusion_noise(flattened_im, num_blocks, size_blocks, im_shape):
    im = flattened_im.reshape(im_shape).copy()
    max_im = max(flattened_im)
    min_im = min(flattened_im)
    for i in range(num_blocks/2):
        r_x = np.random.randint(0, im_shape[0])
        r_y = np.random.randint(0, im_shape[1])
        im[r_x:r_x+size_blocks, r_y:r_y+size_blocks] = max_im

    for i in range(num_blocks/2):
        r_x = np.random.randint(0, im_shape[0])
        r_y = np.random.randint(0, im_shape[1])
        im[r_x:r_x+size_blocks, r_y:r_y+size_blocks] = min_im

    return im.flatten()


def auto_thresh(flattened_im):
    im = flattened_im.copy()
    thr = 0.07; sorted(flattened_im)[int(len(flattened_im)*.65)]
    print thr
    im[where(flattened_im>thr)] = 1
    im[where(flattened_im<=thr)] = 0
    return im


#Download and unpack digits from :
#http://programmingcomputervision.com/downloads/pcv_data.zip

#Change this:
#letters_dir = 'c:/Users/Guerzhoy/Desktop/csc320/pcv_data/a_thumbs/'

im_matrix, im_shape = get_digit_matrix(letters_dir)
for i in range(im_matrix.shape[0]):
    im_matrix[i,:] = im_matrix[i,:]/255.0

V,S,mean_im = pca(im_matrix)


#Show sample "a"s
figure(1)
display_save_25_comps(im_matrix, im_shape)

#Show mean image
figure(2)
imshow(mean_im.reshape(im_shape))

#Show the first 25 principal components
figure(3)
display_save_25_comps(V, im_shape)



##EXAMPLE FOR LECTURE
im = im_matrix[0,:].copy()
n = occlusion_noise(im, 2, 5, im_shape)

figure(4)
imshow(n.reshape(im_shape))

r = get_reconstruction(V[:15,], n,  mean_im)
figure(5)
imshow(r.reshape(im_shape))
r[where(r<0)] = 0


figure(5)

#imshow(r.reshape(im_shape))
figure(6)
imshow(auto_thresh(r).reshape(im_shape))
##############################################



