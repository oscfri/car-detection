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
nPart = 75
maxFilters = 5
# Points that the particle images will be normalized to
pts2 = np.float32([[0,0],[63,0],[0,63],[63,63]])

# Capture first frame

ret, frame = cap.read()
# Our operations on the frame come here
height, width, channels = frame.shape

#extra_area determines how large of a square we will have around the particle, and extra_area of 100 would make a 200 by 200 square
extra_area = 10
# generate x, y coordinates for the particles
w = [np.random.randint(5, 64, size=nPart)]
x = [np.random.randint(extra_area,width-extra_area, size=nPart)]
y = [np.random.randint(extra_area,height-extra_area, size=nPart)]
illegal_windows = [(None, None, None)]

#R is the amount of noise in the predict step
R = 10

#frame_skip is the number of frames we skip in the beginning
frame_index = 0
frame_skip = 0

def draw_rectangle(frame, dimensions, color=(255, 255, 255)):
    x, y, w = dimensions
    cv2.rectangle(frame,(int(x - w),int(y - w)),
                  (int(x + w), int(y + w)),
                  color, 1)

def measure(x, y, w):
    windows = np.zeros((nPart, 64, 64, 3))
    for i in range(nPart):
        #calucalte the coordinates of a square with the particle at the center
        xl = x[i] - w[i]
        xr = x[i] + w[i]
        yd = y[i] - w[i]
        yu = y[i] + w[i]
        
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
    return model.predict(windows)

add_new = True
while(True):
    frame_index += 1
    print frame_index
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame_index < frame_skip:
        continue

    # Our operations on the frame come here
    
    #convert image to gray scale
    #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # We want to add new filter if list of filters are empty
    if not add_new:
        add_new = len(x) == 0
    
    # Add new filter
    if add_new and len(x) < maxFilters:
        w.append(np.random.randint(5, 64, size=nPart))
        x.append(np.random.randint(extra_area,width-extra_area, size=nPart))
        y.append(np.random.randint(extra_area,height-extra_area, size=nPart))
        illegal_windows.append((None, None, None))

    # to_remove are the filters we want to remove after this frame
    # (because the filter has started to become too weak)
    to_remove = []
    add_new = True
    for j in range(len(x)):
        weights = measure(x[j], y[j], w[j])
        for i, weight in enumerate(weights):
            # do not know why x and y are switched but it seems like they are in
            # the right places in the video
            if weight > 0.8:
                draw_rectangle(frame, (x[j][i], y[j][i], w[j][i]),
                               (0, 0, int(weight * 255)))

        w_mean = np.mean(w[j])
        if np.std(x[j]) < w_mean * 2 and np.std(y[j]) < w_mean * 2:
            # Illegal windows, as in windows that no other filter should look
            # into (because this window is already occupied by this filter)
            illegal_windows[j] = (np.mean(x[j]), np.mean(y[j]), w_mean)
            # We want to remove this filter if the filter is too focused on
            # a weak area
            if sum(weights) < nPart * 0.1:
                to_remove.append(j)
                add_new = False
                continue
        else:
            # This filter doesn't focus on a specific area. Bad to define
            # an illegal window
            add_new = False
            illegal_windows[j] = (None, None, None)
        weights = weights / sum(weights)
        #resample
        sampleInds = f.systematic_resample(weights)
        
        #update particles
        x[j] = np.asarray([x[j][i] for i in sampleInds])
        y[j] = np.asarray([y[j][i] for i in sampleInds])
        w[j] = np.asarray([w[j][i] for i in sampleInds])
        
        #predict step, only movement now is noise
        x[j], y[j], w[j] = f.predict(x[j], y[j], w[j], R, width, height,
                                     illegal_windows[:j] + illegal_windows[j+1:])

    for j in range(len(x)):
        if illegal_windows[j][0] is not None:
            draw_rectangle(frame, illegal_windows[j])

    for i, j in enumerate(to_remove):
        x.pop(j - i)
        y.pop(j - i)
        w.pop(j - i)
        illegal_windows.pop(j - i)

    cv2.imshow('frame',frame)
    if frame_index == 20:
        cv2.imwrite('video_20.png', frame)
    if frame_index == 90:
        cv2.imwrite('video_90.png', frame)
    if frame_index == 150:
        cv2.imwrite('video_150.png', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
