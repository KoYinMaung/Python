import cv2
import numpy as np
import keyboard

#call webcam
wCam = cv2.VideoCapture(0);

#Assign ID
id=input('Enter user id = ')
countImg = 0;

#main Body
while(True):
    #read image
    ret,img = wCam.read();

    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, (350,300), (50,50), (0,255,0),2)

    #crop Image with rectangle's Size
    crop_img = img[50:300, 50:350]

    #convert RGB to Gray Image
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY);

    # applying gaussian blur
    value = (25, 25)
    blurred = cv2.GaussianBlur(gray, value, 0)

    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # show thresholded image
    #cv2.imshow('Thresholded', thresh1)

    image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    #Crop Image with Contour Size
    contour_img = gray[y:y+h, x:x+w]
    #contour_img = cv2.resize(gray,(300, 250))

    #For Display Windows
    cv2.imshow("Face", img);
    cv2.imshow('Gray Contour Image', contour_img);

    #wait 5 milisecond for next action
    ch =  cv2.waitKey(5);

    #capture Image press 's' key on keyboard
    if(keyboard.is_pressed('s')):
        countImg = countImg+1;

        #save capture images to dataSet folder
        cv2.imwrite("dataSet/"+str(countImg)+"."+str(id)+".jpg", contour_img)

    #to Stop countImage= 10
    if(countImg == 10): break

    #to Terminal Windows
    if(ch==27): break;      

#delete Cache Memory    
wCam.release()
cv2.destroyAllWindows()
