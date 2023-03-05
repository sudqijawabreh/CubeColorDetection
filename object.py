import cv2
import numpy as np
import math
def show(img):
    cv2.imshow('output',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def GetCubeColor(frame,debug = False):
    greenLower = (29, 86, 6)
    greenUpper = (80, 255, 255)
    lower_red = (0, 25, 21)
    upper_red = (10, 255, 255) 
    lower2 = (160,100,20)
    upper2 = (179,255,255)
    low_blue = (100, 80, 2)
    high_blue = (126, 255, 255)


    #edges = cv2.Canny(img1,100,200)
    #frame = cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
    imgSize = frame.shape[0] * frame.shape[1]

    #edges = cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    if(debug):
        show(frame)
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    th4 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    show(blurred)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    green = getColorMaskArea(frame,hsv,greenLower,greenUpper)
    red = getColorMaskArea(frame,hsv,lower_red,upper_red,lower2,upper2)
    blue = getColorMaskArea(frame,hsv,low_blue,high_blue)
    colors = [('green',green),('blue',blue),('red',red)]

    if(debug):
        print(colors)

    arranged = list(filter(lambda x: x[1] > imgSize * 0.2 and x[1] < imgSize * 0.90,colors))
    arranged = sorted(arranged, key=lambda x : x[1], reverse = True)
    return arranged

def getColorMaskArea(org,img,low1,up1,low2 = None,up2 = None,debug = False):
    org = org.copy()
    if low2 is None:
        low2 = low1
    if up2 is None:
        up2 = up1
    mask1 = cv2.inRange(img, low1, up1)
    mask2 = cv2.inRange(img, low2, up2)
    mask = mask1 + mask2
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(org, contours, -1, (0,255,0), 3)

    if(len(contours) == 0):
        return 0

    #cnt = contours[0]
    #area = cv2.contourArea(cnt)
    #epsilon = 0.1*cv2.arcLength(cnt,True)
    #approx = cv2.approxPolyDP(cnt,epsilon,True)
    show(org)
    ##cv2.drawContours(img, [approx], 0, (0,0,255), 3)
    #dist = math.hypot(x2 - x1, y2 - y1)
    boudningWH = map(lambda x : np.int0(cv2.boxPoints(cv2.minAreaRect(x))),contours)
    #boudningWH = map(lambda x : cv2.boundingRect(x),contours)
    def IsAcceptableRectangle(rect):
        w,h = GetBoxWH(rect)
        #print(w,h)
        if w > 400 and w < 650 and h > 200 and h < 400:
            return True
        else:
            w,h = h,w
            return w > 400 and w < 650 and h > 200 and h < 400

    def GetBoxWH(box):
        x1,y1=(box[0][0],box[0][1])
        x2,y2=(box[1][0],box[1][1])
        x3,y3=(box[2][0],box[2][1])
        x4,y4=(box[3][0],box[3][1])
        dist1 = math.hypot(x2 - x1, y2 - y1)
        dist2 = math.hypot(x3 - x2, y3 - y2)
        return (dist1,dist2)

    
    #boxes = list(boudningWH)
    
    boxes = list(filter(lambda x :IsAcceptableRectangle(x) , boudningWH))


    #x,y,w,h = cv2.boundingRect(cnt)
    #print(w,h)
    if len(boxes) == 0:
        area =  0
    else:
        w,h = GetBoxWH(boxes[0])
        #x,y,w,h = boxes[0]
        #cv2.rectangle(org,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.drawContours(org,[np.int0(boxes[0])],0,(0,0,255),2)
        area =  w*h

    if(debug):
        show(mask)
        show(org)
        print(area)
    return area



def testRed():
    filenames = [ 'red.jpg', 'red_bright.jpg', 'red_bright1.jpg', 'red_dark.jpg', 'red_dark1.jpg' ,'red_rotate1.jpg',  'red_rotate2.jpg']
    for fileName in filenames:
        frame = cv2.imread(fileName)
        values = GetCubeColor(frame)
        if values[0][0] == 'red':
            print(fileName,' passed')
        else:
            print(fileName,' failed')

def testGreen():
    filenames = [
    'green.jpg',
    'green_bright.jpg',
    'green_bright1.jpg',
    'green_dark.jpg',
    'green_redish.jpg',
    'green_saturated.jpg',
    'green_verydark.jpg',
    './green_trim.jpg',
    './green_trimup.jpg',
    './green_trimup_rotate.jpg',
    'green_bluish.jpg',
    'green_bluish1.jpg',
    'green_greenish.jpg',
    ]

    for fileName in filenames:
        frame = cv2.imread(fileName)
        values = GetCubeColor(frame,debug= True)
        if len(values) > 0 and values[0][0] == 'green':
            print(fileName,' passed')
        else:
            print(fileName,' failed')

def test():
    testGreen()

    testRed()
    frame = cv2.imread('./blue.jpg')
    values = GetCubeColor(frame)
    if values[0][0] == 'blue':
        print('blue passed')
    else:
        print('blue failed')
#test()
#testRed()
testGreen()
exit()
#frame = cv2.imread('./blue.jpg')
#print(GetCubeColor(frame)[0])
  
#vid = cv2.VideoCapture(0)
#while True:
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)

import RPi.GPIO as GPIO
# Pin definitions
red = 12
green = 16
blue = 18
nocolor = 22
GPIO.setup(red, GPIO.OUT)
GPIO.setup(green, GPIO.OUT)
GPIO.setup(blue, GPIO.OUT)
GPIO.setup(nocolor, GPIO.OUT)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    #frame = cv2.imread('./green1.jpg')
    frame = frame.array
    #ret, frame = vid.read()
    #print(frame.shape)
    #frame  = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    toshow = cv2.resize(frame,(frame.shape[1]//1,frame.shape[0]//1))
    cv2.imshow('output',toshow)
    rawCapture.truncate(0)
    pressedKey = cv2.waitKey(1) & 0xFF  
    if pressedKey == ord('q'):
        break
    if pressedKey == ord('s'):
        cv2.imwrite('./screenshot.jpg',frame)
        break

    values = GetCubeColor(frame)
    GPIO.output(nocolor, GPIO.LOW)
    GPIO.output(red, GPIO.LOW)
    GPIO.output(green, GPIO.LOW)
    GPIO.output(blue, GPIO.LOW)
        
    if len(values) == 0 :
        print('no color found')
        GPIO.output(nocolor, GPIO.HIGH)
    else:
        print(values[0][0])
        if(values[0][0] == 'green'):
            cv2.imwrite('./green2.jpg',frame)

    if(values[0][0] == 'green'):
        GPIO.output(green, GPIO.HIGH)
    elif(values[0][0] == 'red'):
        GPIO.output(red, GPIO.HIGH)
    elif(values[0][0] == 'blue'):
        GPIO.output(blue, GPIO.HIGH)

    #break
#vid.release()
cv2.destroyAllWindows()
