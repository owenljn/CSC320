import os

###########################################################################
## Handout painting code.
###########################################################################
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import matplotlib.image as mpimg
import scipy as sci
import scipy.misc

###########################################################################
##  Dame this editor, I cannot import canny even they are in the same position
###########################################################################
import os
 
from scipy import *
from scipy.ndimage import *
from scipy.signal import convolve2d as conv

def canny(im, sigma, thresHigh = 50,thresLow = 10):
    '''
        Takes an input image in the range [0, 1] and generate a gradient image
        with edges marked by 1 pixels.
    '''
    imin = im.copy() * 255.0

    # Create the gauss kernel for blurring the input image
    # It will be convolved with the image
    # wsize should be an odd number
    wsize = 5
    gausskernel = gaussFilter(sigma, window = wsize)
    # fx is the filter for vertical gradient
    # fy is the filter for horizontal gradient
    # Please not the vertical direction is positive X

    fx = createFilter([0,  1, 0,
                       0,  0, 0,
                       0, -1, 0])
    fy = createFilter([ 0, 0, 0,
                       -1, 0, 1,
                        0, 0, 0])

    imout = conv(imin, gausskernel, 'valid')
    # print "imout:", imout.shape
    gradxx = conv(imout, fx, 'valid')
    gradyy = conv(imout, fy, 'valid')

    gradx = np.zeros(im.shape)
    grady = np.zeros(im.shape)
    padx = (imin.shape[0] - gradxx.shape[0]) / 2.0
    pady = (imin.shape[1] - gradxx.shape[1]) / 2.0
    gradx[padx:-padx, pady:-pady] = gradxx
    grady[padx:-padx, pady:-pady] = gradyy
    
    # Net gradient is the square root of sum of square of the horizontal
    # and vertical gradients

    grad = hypot(gradx, grady)
    theta = arctan2(grady, gradx)
    theta = 180 + (180 / pi) * theta
    # Only significant magnitudes are considered. All others are removed
    xx, yy = where(grad < 10)
    theta[xx, yy] = 0
    grad[xx, yy] = 0

    # The angles are quantized. This is the first step in non-maximum
    # supression. Since, any pixel will have only 4 approach directions.
    x0,y0 = where(((theta<22.5)+(theta>157.5)*(theta<202.5)
                   +(theta>337.5)) == True)
    x45,y45 = where( ((theta>22.5)*(theta<67.5)
                      +(theta>202.5)*(theta<247.5)) == True)
    x90,y90 = where( ((theta>67.5)*(theta<112.5)
                      +(theta>247.5)*(theta<292.5)) == True)
    x135,y135 = where( ((theta>112.5)*(theta<157.5)
                        +(theta>292.5)*(theta<337.5)) == True)

    theta = theta
    Image.fromarray(theta).convert('L').save('Angle map.jpg')
    theta[x0,y0] = 0
    theta[x45,y45] = 45
    theta[x90,y90] = 90
    theta[x135,y135] = 135
    x,y = theta.shape       
    temp = Image.new('RGB',(y,x),(255,255,255))
    for i in range(x):
        for j in range(y):
            if theta[i,j] == 0:
                temp.putpixel((j,i),(0,0,255))
            elif theta[i,j] == 45:
                temp.putpixel((j,i),(255,0,0))
            elif theta[i,j] == 90:
                temp.putpixel((j,i),(255,255,0))
            elif theta[i,j] == 45:
                temp.putpixel((j,i),(0,255,0))
    retgrad = grad.copy()
    x,y = retgrad.shape

    for i in range(x):
        for j in range(y):
            if theta[i,j] == 0:
                test = nms_check(grad,i,j,1,0,-1,0)
                if not test:
                    retgrad[i,j] = 0

            elif theta[i,j] == 45:
                test = nms_check(grad,i,j,1,-1,-1,1)
                if not test:
                    retgrad[i,j] = 0

            elif theta[i,j] == 90:
                test = nms_check(grad,i,j,0,1,0,-1)
                if not test:
                    retgrad[i,j] = 0
            elif theta[i,j] == 135:
                test = nms_check(grad,i,j,1,1,-1,-1)
                if not test:
                    retgrad[i,j] = 0

    init_point = stop(retgrad, thresHigh)
    # Hysteresis tracking. Since we know that significant edges are
    # continuous contours, we will exploit the same.
    # thresHigh is used to track the starting point of edges and
    # thresLow is used to track the whole edge till end of the edge.

    while (init_point != -1):
        #Image.fromarray(retgrad).show()
        # print 'next segment at',init_point
        retgrad[init_point[0],init_point[1]] = -1
        p2 = init_point
        p1 = init_point
        p0 = init_point
        p0 = nextNbd(retgrad,p0,p1,p2,thresLow)

        while (p0 != -1):
            #print p0
            p2 = p1
            p1 = p0
            retgrad[p0[0],p0[1]] = -1
            p0 = nextNbd(retgrad,p0,p1,p2,thresLow)

        init_point = stop(retgrad,thresHigh)

    # Finally, convert the image into a binary image
    x,y = where(retgrad == -1)
    retgrad[:,:] = 0
    retgrad[x,y] = 1.0
    return retgrad

def createFilter(rawfilter):
    '''
        This method is used to create an NxN matrix to be used as a filter,
        given a N*N list
    '''
    order = pow(len(rawfilter), 0.5)
    order = int(order)
    filt_array = array(rawfilter)
    outfilter = filt_array.reshape((order,order))
    return outfilter

def gaussFilter(sigma, window = 3):
    '''
        This method is used to create a gaussian kernel to be used
        for the blurring purpose. inputs are sigma and the window size
    '''
    kernel = zeros((window,window))
    c0 = window // 2

    for x in range(window):
        for y in range(window):
            r = hypot((x-c0),(y-c0))
            val = (1.0/2*pi*sigma*sigma)*exp(-(r*r)/(2*sigma*sigma))
            kernel[x,y] = val
    return kernel / kernel.sum()
 
def nms_check(grad, i, j, x1, y1, x2, y2):
    '''
        Method for non maximum supression check. A gradient point is an
        edge only if the gradient magnitude and the slope agree

        for example, consider a horizontal edge. if the angle of gradient
        is 0 degress, it is an edge point only if the value of gradient
        at that point is greater than its top and bottom neighbours.
    '''
    try:
        if (grad[i,j] > grad[i+x1,j+y1]) and (grad[i,j] > grad[i+x2,j+y2]):
            return 1
        else:
            return 0
    except IndexError:
        return -1
     
def stop(im, thres):
    '''
        This method is used to find the starting point of an edge.
    '''
    X,Y = where(im > thres)
    try:
        y = Y.min()
    except:
        return -1
    X = X.tolist()
    Y = Y.tolist()
    index = Y.index(y)
    x = X[index]
    return [x,y]
   
def nextNbd(im, p0, p1, p2, thres):
    '''
        This method is used to return the next point on the edge.
    '''
    kit = [-1,0,1]
    X,Y = im.shape
    for i in kit:
        for j in kit:
            if (i+j) == 0:
                continue
            x = p0[0]+i
            y = p0[1]+j

            if (x<0) or (y<0) or (x>=X) or (y>=Y):
                continue
            if ([x,y] == p1) or ([x,y] == p2):
                continue
            if (im[x,y] > thres): #and (im[i,j] < 256):
                return [x,y]
    return -1

###########################################################################


np.set_printoptions(threshold = np.nan)  

def colorImSave(filename, array):
    imArray = scipy.misc.imresize(array, 3., 'nearest')
    if (len(imArray.shape) == 2):
        scipy.misc.imsave(filename, cm.jet(imArray))
    else:
        scipy.misc.imsave(filename, imArray)

def markStroke(mrkd, p0, p1, rad, val):
    # Mark the pixels that will be painted by
    # a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1).
    # These pixels are set to val in the ny x nx double array mrkd.
    # The paintbrush is circular with radius rad>0
    
    sizeIm = mrkd.shape
    sizeIm = sizeIm[0:2];
    nx = sizeIm[1]
    ny = sizeIm[0]
    p0 = p0.flatten('F')
    p1 = p1.flatten('F')
    rad = max(rad,1)
    # Bounding box
    concat = np.vstack([p0,p1])
    bb0 = np.floor(np.amin(concat, axis=0))-rad
    bb1 = np.ceil(np.amax(concat, axis=0))+rad
    # Check for intersection of bounding box with image.
    intersect = 1
    if ((bb0[0] > nx) or (bb0[1] > ny) or (bb1[0] < 1) or (bb1[1] < 1)):
        intersect = 0
    if intersect:
        # Crop bounding box.
        bb0 = np.amax(np.vstack([np.array([bb0[0], 1]), np.array([bb0[1],1])]), axis=1)
        bb0 = np.amin(np.vstack([np.array([bb0[0], nx]), np.array([bb0[1],ny])]), axis=1)
        bb1 = np.amax(np.vstack([np.array([bb1[0], 1]), np.array([bb1[1],1])]), axis=1)
        bb1 = np.amin(np.vstack([np.array([bb1[0], nx]), np.array([bb1[1],ny])]), axis=1)
        # Compute distance d(j,i) to segment in bounding box
        tmp = bb1 - bb0 + 1
        szBB = [tmp[1], tmp[0]]
        q0 = p0 - bb0 + 1
        q1 = p1 - bb0 + 1
        t = q1 - q0
        nrmt = np.linalg.norm(t)
        [x,y] = np.meshgrid(np.array([i+1 for i in range(int(szBB[1]))]), np.array([i+1 for i in range(int(szBB[0]))]))
        d = np.zeros(szBB)
        d.fill(float("inf"))
        
        if nrmt == 0:
            # Use distance to point q0
            d = np.sqrt( (x - q0[0])**2 +(y - q0[1])**2)
            idx = (d <= rad)
        else:
            # Use distance to segment q0, q1
            t = t/nrmt
            n = [t[1], -t[0]]
            tmp = t[0] * (x - q0[0]) + t[1] * (y - q0[1])
            idx = (tmp >= 0) & (tmp <= nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = abs(n[0] * (x[np.where(idx)] - q0[0]) + n[1] * (y[np.where(idx)] - q0[1]))
            idx = (tmp < 0)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q0[0])**2 +(y[np.where(idx)] - q0[1])**2)
            idx = (tmp > nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q1[0])**2 +(y[np.where(idx)] - q1[1])**2)

            #Pixels within crop box to paint have distance <= rad
            idx = (d <= rad)
        #Mark the pixels
        if np.any(idx.flatten('F')):
            xy = (bb0[1]-1+y[np.where(idx)] + sizeIm[0] * (bb0[0]+x[np.where(idx)]-2)).astype(int)
            sz = mrkd.shape
            m = mrkd.flatten('F')
            m[xy-1] = val
            mrkd = m.reshape(mrkd.shape[0], mrkd.shape[1], order = 'F')

            '''
            row = 0
            col = 0
            for i in range(len(m)):
                col = i//sz[0]
                mrkd[row][col] = m[i]
                row += 1
                if row >= sz[0]:
                    row = 0
            '''
            
            
            
    return mrkd

def paintStroke(canvas, x, y, p0, p1, colour, rad):
    # Paint a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1)
    # on the canvas (ny x nx x 3 double array).
    # The stroke has rgb values given by colour (a 3 x 1 vector, with
    # values in [0, 1].  The paintbrush is circular with radius rad>0
    sizeIm = canvas.shape
    sizeIm = sizeIm[0:2]
    idx = markStroke(np.zeros(sizeIm), p0, p1, rad, 1) > 0
    # Paint
    if np.any(idx.flatten('F')):
        canvas = np.reshape(canvas, (np.prod(sizeIm),3), "F")
        xy = y[idx] + sizeIm[0] * (x[idx]-1)
        canvas[xy-1,:] = np.tile(np.transpose(colour[:]), (len(xy), 1))
        canvas = np.reshape(canvas, sizeIm + (3,), "F")
    return canvas

def get_gred(im, sigma):
    '''
        Takes an input image in the range [0, 1] and generate a gradient image
        with edges marked by 1 pixels.
    '''
    imin = im.copy() * 255.0

    # Create the gauss kernel for blurring the input image
    # It will be convolved with the image
    # wsize should be an odd number
    wsize = 5
    gausskernel = gaussFilter(sigma, window = wsize)
    # fx is the filter for vertical gradient
    # fy is the filter for horizontal gradient
    # Please not the vertical direction is positive X

    fx = createFilter([0,  1, 0,
                       0,  0, 0,
                       0, -1, 0])
    fy = createFilter([ 0, 0, 0,
                       -1, 0, 1,
                        0, 0, 0])

    imout = conv(imin, gausskernel, 'valid')
    # print "imout:", imout.shape
    gradxx = conv(imout, fx, 'valid')
    gradyy = conv(imout, fy, 'valid')

    gradx = np.zeros(im.shape)
    grady = np.zeros(im.shape)
    padx = (imin.shape[0] - gradxx.shape[0]) / 2.0
    pady = (imin.shape[1] - gradxx.shape[1]) / 2.0
    gradx[padx:-padx, pady:-pady] = gradxx
    grady[padx:-padx, pady:-pady] = gradyy
    
    # Net gradient is the square root of sum of square of the horizontal
    # and vertical gradients

    grad = hypot(gradx, grady)
    return grad


if __name__ == "__main__":
    # Read image and convert it to double, and scale each R,G,B
    # channel to range [0,1].
    imRGB = array(Image.open('orchid.jpg'))
    imRGB = double(imRGB) / 255.0
    plt.clf()
    plt.axis('off')
    
    sizeIm = imRGB.shape
    sizeIm = sizeIm[0:2]
    # Set radius of paint brush and half length of drawn lines
    rad = 3
    halfLen = 10
    
    # Set up x, y coordinate images, and canvas.
    [x, y] = np.meshgrid(np.array([i+1 for i in range(int(sizeIm[1]))]), np.array([i+1 for i in range(int(sizeIm[0]))]))
    canvas = np.zeros((sizeIm[0],sizeIm[1], 3))
    canvas.fill(-1) ## Initially mark the canvas with a value out of range.
    # Negative values will be used to denote pixels which are unpainted.
    
    # Random number seed
    np.random.seed(29645)
    
    # Orientation of paint brush strokes
    theta = 2 * pi * np.random.rand(1,1)[0][0]
    # Set vector from center to one end of the stroke.
    delta = np.array([cos(theta), sin(theta)])
       
    time.time()
    time.clock()
    '''
    for k in range(500):
        # finding a negative pixel
        # Randomly select stroke center
        cntr = np.floor(np.random.rand(2,1).flatten() * np.array([sizeIm[1], sizeIm[0]])) + 1
        cntr = np.amin(np.vstack((cntr, np.array([sizeIm[1], sizeIm[0]]))), axis=0)
        # Grab colour from image at center position of the stroke.
        colour = np.reshape(imRGB[cntr[1]-1, cntr[0]-1, :],(3,1))
        # Add the stroke to the canvas
        nx, ny = (sizeIm[1], sizeIm[0])
        length1, length2 = (halfLen, halfLen)        
        canvas = paintStroke(canvas, x, y, cntr - delta * length2, cntr + delta * length1, colour, rad)
        #print imRGB[cntr[1]-1, cntr[0]-1, :], canvas[cntr[1]-1, cntr[0]-1, :]
        print ('stroke', k)
    '''
    imGray = 0.3*imRGB[:,:,0] + 0.59*imRGB[:,:,1] + 0.11*imRGB[:,:,2]  
    imCanny = canny(imGray, 4, 10, 1)
    #imCanny = canny(imGray, 2, 30, 10)
    cannyEdge = np.where(imCanny == 1)    
    imGred = get_gred(imGray, 4)
    
    while (-1 in canvas):
        # finding a negative pixel
        # Randomly select stroke center
        center_set = np.where(canvas == -1)
        cntr_index = random.randint(0, (len(center_set[0]) - 1))
        cntr_x = center_set[0][cntr_index]
        cntr_y = center_set[1][cntr_index]
        cntr = [cntr_y, cntr_x]
        print("cntr = ", cntr)
        
        # Orientation of paint brush strokes
        
        theta = pi/2 + imGred[cntr_x][cntr_y] 
        deltaTheta = random.uniform(-15, 15)
        theta = theta + (deltaTheta * pi / 180)
        
        # Set vector from center to one end of the stroke.
        delta = np.array([cos(theta), sin(theta)])
        
        # Grab colour from image at center position of the stroke.
        deltaIntensity = uniform(0.85, 1.15)
        colour = np.reshape(imRGB[cntr[1]-1, cntr[0]-1, :],(3,1))
        
        deltaR = uniform(-15/255, 15/255)
        deltaG = uniform(-15/255, 15/255)
        deltaB = uniform(-15/255, 15/255)
        
        colour[0] = colour[0] + deltaR
        colour[1] = colour[1] + deltaG
        colour[2] = colour[2] + deltaB
        
        if colour[0] > 1.0:
            colour[0] = 1.0
        elif colour[0] < 0:
            colour[0] = 0.0
        if colour[1] > 1.0:
            colour[1] = 1.0
        elif colour[1] < 0:
            colour[1] = 0.0
        if colour[2] > 1.0:
            colour[2] = 1.0
        elif colour[2] < 0:
            colour[2] = 0.0
        
        # Add the stroke to the canvas
        nx, ny = (sizeIm[1], sizeIm[0])
        length1, length2 = (halfLen, halfLen)
        
        l1set, l2set = (0, 0)
        l1, l2 = (halfLen, halfLen)
        for r in range(halfLen):
            if ((l1set == 1) and (l2set == 1)):
                break
            else:
                if (l1set == 0):
                    l1current = cntr + delta * r
                    try:
                        isedge1 = imCanny[int(l1current[1])][int(l1current[0])]
                    except:
                        if (l1set == 0):
                            isedge1 = 1
                        else:
                            isedge1 = 0
                    if ((isedge1 == 1) or (r == halfLen)):
                        l1 = r
                        l1set = 1
                if (l2set == 0):
                    l2current = cntr - delta * r
                    try:
                        isedge2 = imCanny[int(l2current[1])][int(l2current[0])]
                    except:
                        if (l2set == 0):
                            isedge2 = 1
                        else:
                            isedge2 = 0
                    if ((isedge2 == 1) or (r == halfLen)):
                        l2 = r
                        l2set = 1
        length1, length2 = (l1, l2)
        #print("from ", l1, " to ", l2)
        rad = randint(1, 3)
        cntr[0] = cntr[0] + 1
        cntr[1] = cntr[1] + 1
        canvas = paintStroke(canvas, x, y, cntr - delta * length2, cntr + delta * length1, colour, rad)
        #print imRGB[cntr[1]-1, cntr[0]-1, :], canvas[cntr[1]-1, cntr[0]-1, :]
    
    
    
    print ("done!")
    time.time()
    
    canvas[canvas < 0] = 0.0
    plt.clf()
    plt.axis('off')
    plt.imshow(canvas)
    plt.pause(3)
    colorImSave('output.png', canvas)
    plt.close()
    imshow(canvas)
    show()
