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
cap = cv2.VideoCapture('../road.mp4')

# Set number of particles
nPart = 200
# set number of random particles introduced each resample
nrandPart = 40
# Points that the particle images will be normalized to
pts2 = np.float32([[0,0],[63,0],[0,63],[63,63]])

# Capture first frame                                                               \
                                                                                         
ret, frame = cap.read()
# Our operations on the frame come here
height, width, channels = frame.shape

#extra_area determines how large of a square we will have around the particle, and extra_area of 100 would make a 200 by 200 square
extra_area = 20
# generate x, y coordinates for the particles                                           
x = np.random.randint(extra_area,width-extra_area, size=nPart)
y = np.random.randint(extra_area,height-extra_area, size=nPart)

#R is the amount of noise in the predict step
R = 10

#frame_skip is the number of frames we skip in the beginning
frame_index = 0
frame_skip = 0

while(True):
    frame_index += 1
    # Capture frame-by-frame                                                                                       
    ret, frame = cap.read()
    if frame_index < frame_skip:
        continue

    # Our operations on the frame come here
    
    #convert image to gray scale
    #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    weights = np.ones((nPart, 1))
    windows = np.zeros((nPart, 64, 64, 3))
    
    for i in range(nPart):
        #calucalte the coordinates of a square with the particle at the center
        
        xl = x[i]-extra_area
        xr = x[i]+extra_area
        yd = y[i]-extra_area
        yu = y[i]+extra_area
        
        #transform this square to 64x64
        pts1 = np.float32([[xl,yd],[xr,yd],[xl,yu],[xr,yu]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(frame,M,(64,64))
        
        # Store the current window
        windows[i, :, :, :] = dst[:64, :64, :] / 255.0

        #calculate the weights, I don't think this is right
        #scale = model.predict_proba(dst.reshape((1,64,64,1)))
        #weights.append(scale)


    # Calculate all weights in one go
    weights = model.predict(windows)

    for i, weight in enumerate(weights):
        # do not know why x and y are switched but it seems like they are in
        # the right places in the video
        #cv2.rectangle(frame,(int(x[i] - extra_area),int(y[i] - extra_area)),
         #             (int(x[i] + extra_area), int(y[i] + extra_area)),
          #            (0,0, int(weight * 255)), 1)
        if weight > 0.85:
            cv2.circle(frame,(int(x[i]),int(y[i])), 2, (0,0,255), -1)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow('frame',windows[nPart-1, :, :, :])
    #normalize the weights
    weights = weights / sum(weights)
    #resample
    sampleInds = f.multinomial_resample(weights,nPart-nrandPart)
    
    #update particles
    x = np.asarray([x[i] for i in sampleInds])
    y = np.asarray([y[i] for i in sampleInds])
    
    x = np.concatenate((x,np.asarray(np.random.randint(extra_area,width-extra_area,size=nrandPart))),axis=0)
    y = np.concatenate((y,np.asarray(np.random.randint(extra_area,height-extra_area,size=nrandPart))),axis=0)
    #predict step, only movement now is noise
    
    x,y = f.predict(x,y,R)

    x = np.clip(x, extra_area, width - extra_area)
    y = np.clip(y, extra_area, height - extra_area)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture                                                                        
cap.release()
cv2.destroyAllWindows()
