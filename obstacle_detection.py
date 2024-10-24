#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


# In[3]:


def max_dist_pix(list_of_arr):
    dist_arrays = [distance(arr) for arr in list_of_arr]
    max_dist = max(dist_arrays)#dist(arr1), dist(arr2), dist(arr3)
    return max_dist


# In[4]:


def line(coeff, n):
    coord = []
    for i in range(n):
        coord.append(np.rint(coeff*i))
    return coord

def make_line_values(image_array, coeff, n):
    arr = line(coeff,n)
    array = np.zeros([len(arr)])
    for n in range(len(arr)):
        idx = arr[n]+[x,y]
        new_x = int(idx[0])
        new_y = int(idx[1])
        array[n] = image_array[new_x,new_y]
    return array


# In[5]:


def min_dist(pix_dist_max):
    return 3900/pix_dist_max #36 - coeff empiric


# In[48]:


def distance(data):
    mean_value = np.mean(data)
    indices_greater_than_mean = np.where(data > (min([mean_value+30, 240])))[0]
    n = len(indices_greater_than_mean)
    # Sum the indices and find center
    dist = np.sum(indices_greater_than_mean)/n
    return dist


# In[19]:


def frame_process(frame, k):
    
    arr1 = make_line_values(frame, coeff1, k)
    arr2 = make_line_values(frame, coeff2, k)
    arr3 = make_line_values(frame, coeff3, k)
    list_of_arr = [arr2, arr1, arr3]

    pix_dist_max = max_dist_pix(list_of_arr)
    #print(f'pix_dist_max - {pix_dist_max}')
    min_distance = min_dist(pix_dist_max)
    
    return min_distance


# In[36]:


def draw_img(image, distance):
    draw = ImageDraw.Draw(image)

    # Define the text and font
    text = f'distance to objects - {distance} cm'
    font_size = 50
    font = ImageFont.load_default()  # You can also use a specific TTF font file

    # Define the position and color
    position = (50, 50)  # Change this to your desired position
    color = 128  # White color in RGB

    # Add text to the image
    draw.text(position, text, fill=color, font=font)
    return image


# In[39]:


def inform_drone():
    print(f'obstacle ahead, distance - {dist}')


# In[45]:


coeff1 = np.array([-1, -0.115])
coeff2 = np.array([1, -0.2051])
coeff3 = np.array([0.1541, 1])
y = 287
x = 278
k = 50 #num of pixels from center max
treshold = 800 #minimum dist to inform drone 


# In[49]:


path = os.getcwd()
cap = cv2.VideoCapture(1)  # Open the default camera (0)
count = 0
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale

    # Perform processing here on 'frame'
    #cv2.imwrite(f"{path}/frames/frame{count}.jpg", frame)     # save frame as JPEG file
    
    dist = frame_process(frame, k)
    #print(dist)
    if dist < treshold:
        inform_drone
    #write distance to frame
    
    
    img= Image.fromarray(frame, 'L')
    image = draw_img(img, dist)
    
    #image.save(f'{path}/output/output_image_{count}.jpg')
    count +=1
    frame = np.array(image)

    cv2.imshow('Webcam', frame)  # Display the frame
    

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




