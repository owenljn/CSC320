
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
from scipy.ndimage import filters
import urllib
import Image



act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']





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
    