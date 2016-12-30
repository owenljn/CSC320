
from pylab import *
import numpy as np
import random
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from scipy.ndimage import filters
import urllib
import Image


#%matplotlib

act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']

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
    im_files = sorted([img_dir + filename for filename in os.listdir(img_dir) if filename[-4:] == ".jpg"])
    im_shape = array(imread(im_files[0])).shape[:2] # open one image to get the size 
    im_matrix = array([imread(im_file)[:,:,0].flatten() for im_file in im_files])
    im_matrix = array([im_matrix[i,:]/(norm(im_matrix[i,:])+0.0001) for i in range(im_matrix.shape[0])])
    return (im_matrix, im_shape, im_files)
    

def get_reconstruction(V, im, mean_im):
    coefs = [np.dot(V[i,:], (im-mean_im)) for i in range(V.shape[0])]
    new_im = mean_im.copy()
    for i in range(len(coefs)):
        new_im = new_im + coefs[i]*V[i, :]
    return new_im

def auto_thresh(flattened_im):
    im = flattened_im.copy()
    thr = 0.07; sorted(flattened_im)[int(len(flattened_im)*.65)]
    print(thr)
    im[where(flattened_im>thr)] = 1
    im[where(flattened_im<=thr)] = 0
    return im



def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def crop(im):
    while (max(shape(im)) > 64):
        im = imresize(im, 0.5)
    im = imresize(im, [32,32])
    output = 0.3*im[:,:,0] + 0.59*im[:,:,1] + 0.11*im[:,:,2]
    return output/255



def part1():
    testfile = urllib.URLopener()
    
    #Note: you need to create the uncropped folder first in order 
    #for this to work
    
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open("/h/u12/g4/00/g3yangyl/Downloads/320/a3/faces_subset.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "/h/u12/g4/00/g3yangyl/Downloads/320/a3/uncropped/"+filename), {}, 30)
                print (filename)
                if not os.path.isfile("/h/u12/g4/00/g3yangyl/Downloads/320/a3/uncropped/"+filename):
                    continue
                else:
                    try:
                        x1,y1,x2,y2 = line.split()[5].split(",")
                        im = imread("/h/u12/g4/00/g3yangyl/Downloads/320/a3/uncropped/"+filename)
                        bbim = im[int(y1):int(y2), int(x1):int(x2)]
                        cropim = crop(bbim)
                        imsave("/h/u12/g4/00/g3yangyl/Downloads/320/a3/cropped/"+filename, cropim)
                    except:
                        continue
                print (filename + " cropped...")
                i += 1

def part3(V, im, mean_im, k, im_files, im_coefs):
    coefs = [np.dot(V[i,:], (im - mean_im)) for i in range(V.shape[0]-3)]
    interval = range(k)
    min_distance = -1
    # choose a sub coefs based on k
    chosen_coefs = []
    for j in interval:
        chosen_coefs.append(coefs[j])
    # check and compare all the projection of training image
    for i in range(im_coefs.shape[0]):
        curr_coefs = im_coefs[i]
        # choose a sub coefs based on k
        chosen_curr_coefs = []
        for j in interval:
            chosen_curr_coefs.append(curr_coefs[j])
        # culculate the distance
        distance = sqrt(((array(chosen_curr_coefs) - array(chosen_coefs)) ** 2).sum())
        # update the minimum distance
        if(min_distance == -1 or distance < min_distance):
            min_distance = distance
            min_img = i # the index of nearest training image
        # return the guessed person's name
    output = rm_number(im_files[min_img].split('/')[-1][:-4])
    file_name = (im_files[min_img].split('/')[-1])
    return output, file_name

def part4(V, im, mean_im, k, im_files, im_coefs):
    coefs = [np.dot(V[i,:], (im - mean_im)) for i in range(V.shape[0]-3)]
    interval = range(k)
    min_distance = -1
    # choose a sub coefs based on k
    chosen_coefs = []
    for j in interval:
        chosen_coefs.append(coefs[j])
    # check and compare all the projection of training image
    for i in range(im_coefs.shape[0]):
        curr_coefs = im_coefs[i]
        # choose a sub coefs based on k
        chosen_curr_coefs = []
        for j in interval:
            chosen_curr_coefs.append(curr_coefs[j])
        # culculate the distance
        distance = sqrt(((array(chosen_curr_coefs) - array(chosen_coefs)) ** 2).sum())
        # update the minimum distance
        if(min_distance == -1 or distance < min_distance):
            min_distance = distance
            min_img = i # the index of nearest training image
        # return the guessed person's name
    output = rm_number(im_files[min_img].split('/')[-1][:-4])
    return output

def rm_number(s):
    return ''.join([i for i in s if not i.isdigit()])

def ismale(name):
    return ((name == "brody") or (name == "eckhart") or (name == "sankler"))


####################################
###             Main             ###
####################################

training_dir = '/h/u12/g4/00/g3yangyl/Downloads/320/a3/part2/training/'
test_dir = '/h/u12/g4/00/g3yangyl/Downloads/320/a3/part2/test/'

#Part 2
im_matrix, im_shape, im_files = get_digit_matrix(training_dir)
test_matrix, test_shape, test_files = get_digit_matrix(test_dir)
for i in range(im_matrix.shape[0]):
    im_matrix[i,:] = im_matrix[i,:]/255.0
for i in range(test_matrix.shape[0]):
    test_matrix[i,:] = test_matrix[i,:]/255.0
V,S,mean_im = pca(im_matrix)
print("pca done...")
'''
imsave('/h/u12/g4/00/g3yangyl/Downloads/320/a3/part2/immean.jpg',mean_im.reshape(im_shape))
i = 0
while i < 25:
    imsave('/h/u12/g4/00/g3yangyl/Downloads/320/a3/part2/eigen'+str(i)+'.jpg',V[i].reshape(im_shape))
    i = i + 1
'''

im_coefs = []
for j in range(0, im_matrix.shape[0]):
    im_coefs.append([np.dot(V[i,:], (im_matrix[j] - mean_im)) for i in range(V.shape[0]-3)])
im_coefs = array(im_coefs)
print("init done...")
'''
#Part 3
k_list = [2, 5, 10, 20, 50, 80, 100, 150, 200]
example_counter = 0
for k in k_list:
    total = 0
    hit = 0
    for i in range(test_matrix.shape[0]):
        total += 1
        true_name = rm_number(test_files[i].split('/')[-1][:-4])
        hit_name, fn = part3(V, test_matrix[i], mean_im, 150, im_files, im_coefs)
        if (true_name == hit_name):
            hit += 1
            #print("hit")
        else:
            example_counter += 1
            if (example_counter <= 5):
                print(fn, test_files[i].split('/')[-1])
            #print("miss:" + hit_name)
    hit_rate = double(hit) / double(total)
    print("current k:" + str(k))
    print("hit_rate = " + str(hit_rate))
"""
pca done...
init done...
current k:2
hit_rate = 0.189189189189
current k:5
hit_rate = 0.391891891892
current k:10
hit_rate = 0.527027027027
current k:20
hit_rate = 0.567567567568
current k:50
hit_rate = 0.581081081081
current k:80
hit_rate = 0.594594594595
current k:100
hit_rate = 0.621621621622
current k:150
hit_rate = 0.635135135135
current k:200
hit_rate = 0.635135135135

failure cases:
('anders38.jpg', 'agron111.jpg')
('anders98.jpg', 'agron113.jpg')
('benson33.jpg', 'agron115.jpg')
('agron22.jpg', 'anders100.jpg')
('applegate23.jpg', 'anders109.jpg')
"""
'''
# Part4
total = 0
hit = 0
k_list = [2, 5, 10, 20, 50, 80, 100, 150, 200]
example_counter = 0
for k in k_list:
    for i in range(test_matrix.shape[0]):
        total += 1
        true_name = rm_number(test_files[i].split('/')[-1][:-4])
        hit_name = part4(V, test_matrix[i], mean_im, k, im_files, im_coefs)
        if (ismale(hit_name) == ismale(true_name)):
            hit += 1
            #print("hit")
        #else:
            #print("miss:" + hit_name)
    hit_rate = double(hit) / double(total)
    print("current k:" + str(k))
    print("hit_rate = " + str(hit_rate))
"""
pca done...
init done...
current k:2
hit_rate = 0.635135135135
current k:5
hit_rate = 0.662162162162
current k:10
hit_rate = 0.720720720721
current k:20
hit_rate = 0.75
current k:50
hit_rate = 0.781081081081
current k:80
hit_rate = 0.801801801802
current k:100
hit_rate = 0.818532818533
current k:150
hit_rate = 0.831081081081
current k:200
hit_rate = 0.840840840841
"""




