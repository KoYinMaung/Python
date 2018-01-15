import cv2
import numpy as np
import glob
import os

#call webcam
cap = cv2.VideoCapture(0)

#main Body 
while(cap.isOpened):
    #read Image
    ret, img = cap.read()

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

    #calculate contours
    image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 1)

    #Crop Image with Contour Size
    contour_img = gray[y:y+h, x:x+w]
    #contour_img = cv2.resize(gray, (300,250))

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    #call template Image Dataset
    #template_data=[]
    mypath= glob.glob('dataSet\\*.jpg')

    #matching All Image 
    for file in mypath:
        #split FileName and FileType
        filename, ext = os.path.splitext(file)
        
        #get only FileName 
        filename = filename[filename.rfind("\\")+3:]

        #read matching image with gray
        image = cv2.imread(file, 0)

        #matching with Method
        #cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED
        res = cv2.matchTemplate(contour_img, image,cv2.TM_CCOEFF_NORMED)
        #template_data.append(image)    

        #matching with Threshold Vale
        threshold = 0.6

        #matching Image location by Threshold
        loc = np.where(res >= threshold)

        #Display Text with specified location
        for pt in zip(*loc[::-1]):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(crop_img, filename, (20,50), font, 1.5,(255,10,10),2,cv2.LINE_AA)
    #print(filename)

    #For View Windows
    cv2.imshow('Main Image', img)
    cv2.imshow('Gray Contour Image', contour_img);
    #cv2.imshow('Contours Image', drawing)
    #all_img = np.hstack((drawing, contour_img))
    #cv2.imshow('Contours', all_img)

    #For Terminal Windows
    key = cv2.waitKey(10)
    if(key==27):
        break


