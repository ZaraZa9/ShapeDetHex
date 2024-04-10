import cv2
import numpy
import math

#WHAT THE MAIN PURPOSE IS
#Easiest way to determine a PERFECT SHAPE is to count the points / edges
#Does not work if image is a bit wonky - lets say if trangles image is of low res
#Thereshholding it would make the side of the traingle "bumpy" and then
#cv2 would detect the bums as possible edges - this can be countered by
#fine tuning some parameters however this is not the most reliable way of doing
#so, hence as addition to tge counting of edges I've implemented a masking algorithm
#   1.To identify the mask to be used - first determine the area of all the potential shapes enclosed in the detected points 
#   2.Identfy the area enclosed by the points of the shape
#   3.Deterime the possible masks ot be used 
#which
#   1. identifies which mask is more appropriate to use
#   2. RESIZES the mask to an image
#       a. get the longest side of of the original image - most likely to be one of its sides
#       b. resize the mask to match the length of the side
#   3. [done] Get percentage coverage of the resized maks over an original image
#   4. based on percentage coverage - determine the shape
#       a. if the percentage coverage is relatively low - try another mask

shape = ['CIRCLE', 'SEMICIRCLE', 'QUARTER CIRCLE', 'TRIANGLE', 'RECTANGLE', 'PENTAGON', 'STAR', 'CROSS', '*']

frameWidth = 500

img = cv2.imread('image0.jpg')
img2 = cv2.imread('squareMask.png',cv2.IMREAD_GRAYSCALE)
imgContour = img.copy()

def nothing(x):
    pass

#cv2.namedWindow("Parameters")
#cv2.resizeWindow("Parameters",640,240)
#cv2.createTrackbar("Threshold1","Parameters",150,255,nothing)
#cv2.createTrackbar("Threshold2","Parameters",255,255,nothing)

approx = []
def getContours(img,imgContour):
    #RETURNS APPROXIMATED POINT COORDINATES IN THE ORDER THEY ARE DETECTED ON THESHAPE, i.e. POINTS FOLLOW ONE AFTER THE OTHER DRAWING OUT THE PERIMETER OF THE SHAPE
    #POINTS ARE FOLLOWING ONE AFTER THE OTHER AS THEY WOULD BE DETECTED ON THE PERIMETER OF THE SHAPE
    #i.e. [[[273 247]] -> [[308 363]] -> [[430 363]] -> [[464 246]] -> [[365 180]]] draws out a pentagon starting from point (273,247) and ending with (365,180)

    #NOW I DO NOT NEED TO CHEKC FOR THE LARGEST POSSIBLE DISTANCE BETWEEN THEPOINTS
    #CHECK FOR THE LONGEST SIDE, I.E. [[[273 247]] -> [[308 363]] & [[308 363]] -> [[430 363]]
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(imgContour, contours, -1, (255,0,0),4)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            cv2.drawContours(imgContour, cnt, -1,(255,0,255),5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            #array of points (their coordinates)
            print(len(approx))
            print(approx)
            points = numpy.array(approx)
            mean_position = (numpy.mean(points, axis=0)).flatten()
            print("Mean Position:", mean_position)
            cv2.rectangle(imgContour,(int(mean_position[0]), int(mean_position[1])),(int(mean_position[0])+5, int(mean_position[1])+5),(0,255, 0),5)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x, y),(x+w, y+h,),(0,255, 0),5)
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x, y - 50), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
            return approx, mean_position, area


def getGreatestSide(arr):
    points = arr.reshape(-1, 2)
    greatest = 0

    x1 = points[0][1]
    x2 = points[len(points)-1][1]
    y1 = points[0][0]
    y2 = points[len(points)-1][0]
    greatest = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print(greatest)
    for i in range (len(points)-1):
        x1 = points[i][1]
        x2 = points[i+1][1]
        y1 = points[i][0]
        y2 = points[i+1][0]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        print(distance)
        if greatest < distance:
            greatest = distance
    print (f"Longest side {greatest}")
    return greatest


def getPercentageCoverage(centrePoint, image, maskImg):
    angle = 10  # Change this angle as needed
    currentAngle = 0

    while currentAngle != 370:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        centre = (int(centrePoint[0]), int(centrePoint[1]))
        mask_image = cv2.resize(maskImg, (image.shape[1], image.shape[0]))
        cv2.imshow("ResizedMask",mask_image)

        height, width = mask_image.shape[:2]

        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), currentAngle, 1.0)

        rotated_image = cv2.warpAffine(mask_image, rotation_matrix, (width, height))

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=rotated_image)
        cv2.imshow('Result2', result)

        white_pixels_of_result = cv2.countNonZero(result)
        white_pixels_of_initial = cv2.countNonZero(image)
        
        percentage_coverage = (white_pixels_of_result / white_pixels_of_initial) * 100
        print(f"Percentage Coverage at angle {currentAngle} : {percentage_coverage:.2f}%")
        currentAngle += angle


   
        

        
imgCanny = cv2.Canny(img,20,30)
kernel = numpy.ones((5,5))
imgDil = cv2.dilate(imgCanny,kernel, iterations=1)

approx, centrePoint, areaOfShape = getContours(imgDil, imgContour)


sideLength = getGreatestSide(approx)


maksShapeSideLength = 100
#scaleFactor = sideLength / maksShapeSideLength
#img2 = cv2.resize(img2, (0, 0), fx=scaleFactor, fy=scaleFactor)

print(f"Test image size 1 {img.shape[1]} , {img.shape[0]}")
print(f"Test image size 2 {img2.shape[1]} , {img2.shape[0]}")

scaleFactor = maksShapeSideLength / sideLength
img = cv2.resize(img, (0, 0), fx=scaleFactor, fy=scaleFactor)

print(f"Test image size 1 {img.shape[1]} , {img.shape[0]}")
print(f"Test image size 2 {img2.shape[1]} , {img2.shape[0]}")

#cv2.imshow("ResizedMask",img2)
cv2.imshow("ResizedOGImage",img)


# Example usage
getPercentageCoverage(centrePoint, img, img2)



cv2.imshow("Result", imgContour)
cv2.waitKey(0)
cv2.destroyAllWindows()
