import numpy as np
import cv2
from random import randint
from keras.models import load_model
import load_images
import filtFuncs as f                                                         
import time

# load model
model = load_model("car_model.h5")
# load video
cap = cv2.VideoCapture('../DrivingaroundDublinCityDashcam.mp4')

# Set number of particles
nPart = 100
# Points that the particle images will be normalized to
pts2 = np.float32([[0,0],[63,0],[0,63],[63,63]])

# Capture first frame                                                               \
                                                                                         
ret, frame = cap.read()
# Our operations on the frame come here                                                                                                                                          
height, width, channels = frame.shape


#extra_area determines how large of a square we will have around the particle, and extra_area of 100 would make a 200 by 200 square
extra_area = 100
# generate x, y coordinates for the particles                                           
x = [randint(extra_area,(height-extra_area)) for p in range(1,nPart)]
y = [randint(extra_area,(width-extra_area)) for p in range(1,nPart)]


#R is the amount of noise in the predict step
R = 10

while(True):
    # Capture frame-by-frame                                                                                       
    ret, frame = cap.read()
    # Our operations on the frame come here                                                                                                                                        
    
    #convert image to gray scale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    weights =[]
    
    for i in range(0,nPart-1):
        #calucalte the coordinates of a square with the particle at the center
        
        xl = x[i]-extra_area
        xr = x[i]+extra_area
        yd = y[i]-extra_area
        yu = y[i]+extra_area
        
        #transform this square to 64x64
        pts1 = np.float32([[xl,yd],[xr,yd],[xl,yu],[xr,yu]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(gray_image,M,(64,64))
        
        #calculate the weights, I don't think this is right
        scale = model.predict_proba(dst.reshape((1,64,64,1)))
        weights.append(scale)

        # do not know why x and y are switched but it seems like they are in
        # the right places in the video
        cv2.circle(frame,(int(y[i]),int(x[i])), 2, (0,0,255), -1)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    #normalize the weights
    sumW = sum(weights)
    weights[:] = [i/sumW for i in weights]
    #resample
    sampleInds = f.systematic_resample(weights)
    
    #update particles
    x = [x[i] for i in sampleInds]
    y = [y[i] for i in sampleInds]
    
    #predict step, only movement now is noise
    x,y = f.predict(x,y,R)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture                                                                        
cap.release()
cv2.destroyAllWindows()
