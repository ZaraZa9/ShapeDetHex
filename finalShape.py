import cv2
import numpy
import math
import time

# Record the start time
start_time = time.time()

#1. Find number of points and area of the shape
#2. Find the largest side of a shape
#3. Re-scale the area of the mask by the factor of (side of the image/side of the mask shape)^2
#4. Look up which maks is the most appropriate mask to use
#5. Perform a bitewise operation and get percentage coverage
#6. Based on percentage coverage - if the mask deos not reach sutisfyable coverage - use the mask which has area lower / higher of that used before




#1. Find number of points

sample = cv2.imread('Filtered/GreenSemicircleWhiteW_3.png')
#sample = cv2.imread('image0.jpg')
#sample = cv2.imread('ActualMasks/crossMask.png')
#sample = cv2.imread('ActualMasks/crossMasktest.jpg')

if sample is None:
    print("Error: Unable to load the image.")
    exit()

imgContour = sample.copy()

def nothing(x):
    pass

approx = []
def getContours(img,imgContour):
    #RETURNS APPROXIMATED POINT COORDINATES IN THE ORDER THEY ARE DETECTED ON THESHAPE, i.e. POINTS FOLLOW ONE AFTER THE OTHER DRAWING OUT THE PERIMETER OF THE SHAPE
    #POINTS ARE FOLLOWING ONE AFTER THE OTHER AS THEY WOULD BE DETECTED ON THE PERIMETER OF THE SHAPE
    #i.e. [[[273 247]] -> [[308 363]] -> [[430 363]] -> [[464 246]] -> [[365 180]]] draws out a pentagon starting from point (273,247) and ending with (365,180)

    #NOW I DO NOT NEED TO CHEKC FOR THE LARGEST POSSIBLE DISTANCE BETWEEN THEPOINTS
    #CHECK FOR THE LONGEST SIDE, I.E. [[[273 247]] -> [[308 363]] & [[308 363]] -> [[430 363]]
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            cv2.drawContours(imgContour, cnt, -1,(255,0,255),5)
            peri = cv2.arcLength(cnt, True)
            #approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #array of points (their coordinates)
            print(len(approx))
            print(approx)
            points = numpy.array(approx)
            mean_position = (numpy.mean(points, axis=0)).flatten()
            print("Mean Position: ", mean_position)
            cv2.rectangle(imgContour,(int(mean_position[0]), int(mean_position[1])),(int(mean_position[0])+5, int(mean_position[1])+5),(0,255, 0),5)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x, y),(x+w, y+h,),(0,255, 0),5)
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x, y - 50), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
            return approx, mean_position, area, x, y, w, h

imgCanny = cv2.Canny(sample,20,30)
kernel = numpy.ones((5,5))
imgDil = cv2.dilate(imgCanny,kernel, iterations=1)
cv2.imshow("Dilated",imgDil)
cv2.imshow("imgContour",imgContour)
approx, centrePoint, areaOfShape, x, y, w, h = getContours(imgDil, imgContour)
print(f"Area of the shape: {areaOfShape}")
cv2.imshow("Result", imgContour)


#2. Find the largest side of a shape

def getGreatestSide(arr):
    points = arr.reshape(-1, 2)
    greatest = 0
    gr = None
    for i in range(len(points)-1):
        distance = math.dist(points[i], points[i+1])
        if distance > greatest:
            greatest = distance
            gr = (points[i], points[i+1])
        
    print(f"Longest side: {greatest}")
    return greatest, gr

sideLength, gLine = getGreatestSide(approx)
cv2.line(imgContour, tuple(gLine[0]), tuple(gLine[1]), (0, 0, 255), 3)
cv2.imshow('Image with Line', imgContour)
cv2.waitKey(0)


#3. Re-scale the area of the mask by the factor of (side of the image/side of the mask shape)^2
'''
shapes = [['RECTANGLE', 10000, 100,"testMasks/squareMask.png"],
          ['PENTAGON', 7700, 65,"testMasks/pentagonMask.png"],
          ['TRIANGLE', 5800, 110,"testMasks/triangleMask.png"],
          ['CIRCLE', 8500, 40,"testMasks/circleMask.png"]]
'''

shapes = [['RECTANGLE', 10000, 100,"ActualMasks/squareMask.png"],
          ['PENTAGON', 7700, 65,"ActualMasks/pentagonMask.png"],
          ['TRIANGLE', 5800, 110,"ActualMasks/triangleMask.png"],
          ['CIRCLE', 8500, 40,"ActualMasks/circleMask.png"],
          ['HALF_CIRCLE',4600,100,"ActualMasks/halfCircleMask.png"],
          ['STAR',5100,67,'ActualMasks/starMask.png'],
          ['CROSS',5200,37,"ActualMasks/crossMask.png"],
          ['QUATER_CIRCLE',7800,100,"ActualMasks/quaterCircleMask.png"]]



# Rescale the area of the mask

#THERE ARE 2 LOOPS HERE - OPTIMISE THIS PLEASE
for  shape in shapes:
    shape[1] = shape[1] * (sideLength / shape[2]) ** 2
    print(f"New area of shape {shape[0]} is {shape[1]}")


# Calculate absolute differences - 2ND LOOP HERE
differences = {shape[0]: abs(shape[1] - areaOfShape) for shape in shapes}




#4. Look up which maks is the most appropriate mask to use

closest_shape = min(differences, key=differences.get) # Find the shape with the smallest absolute difference

threshold = 500 # Set a threshold for an educated guess (you can adjust this value)

# Check if the smallest difference is below the threshold
if differences[closest_shape] < threshold:
    print(f"The shape that closely resembles the given area is {closest_shape}")
else:
    print(f"No exact match found. An educated guess could be {closest_shape}")

path = next((shape[3] for shape in shapes if closest_shape == shape[0]), None) # Get the path of the first shape matching closest_shape, or None if not found


mask = cv2.imread(path)

print(f"Path: {path}")
cv2.imshow("Selected Mask: ",mask)





#5. Perform a bitewise operation and get percentage coverage

#focuses in on the are of interest (the green bounding box)
focusedImage = sample[y:y+h,x:x+w]
#cv2.imshow("Focused image",focusedImage)

#So now what I would want to do is to 
#RESIZE THE MASK IMAGE (sideLength / shape[2] of the selected shape - how to store it find out yourself)
#rotate the mask around the centre point specified 
#IF SOME BS HAPPENS AND IT DOESNT LIKE THAT THE SIZES OF IAMGE AND MASK ARE NOT EXACT - EXTEND THE TRANSPARENT PART OF THE MASK TO COVER THE WHOLE SAMPLE IMAGE AND TELL IT TO FUCK OFF
#get percentage coverage and we're done

#ratio = sideLength / mask.shape[1]

# Calculate the new dimensions based on the ratio
#new_width = int(mask.shape[1] * ratio)
#new_height = int(mask.shape[0] * ratio)

# Resize the image
#mask = cv2.resize(mask, (new_width, new_height))
#cv2.imshow("Resized mask ",mask)

image_with_transparency = mask
# Set the desired size for the extended canvas

cv2.imshow("Here0",mask)

desired_height = mask.shape[0]
desired_width = mask.shape[1]
if mask.shape[0] < focusedImage.shape[0]:
    desired_height = focusedImage.shape[0]
if mask.shape[0] > focusedImage.shape[0]:
    mask = cv2.resize(mask, (mask.shape[1], focusedImage.shape[0]))
    desired_height = focusedImage.shape[0]

cv2.imshow("Here1",mask)
cv2.waitKey(0)

if mask.shape[1] <= focusedImage.shape[1]:
    desired_width = focusedImage.shape[1]
if mask.shape[1] > focusedImage.shape[1]:
    mask = cv2.resize(mask, (focusedImage.shape[1], mask.shape[0]))
    desired_width = focusedImage.shape[1]

cv2.imshow("Here2",mask)
cv2.waitKey(0)

# Calculate the position where the original image will be placed on the new canvas
position_y = round((desired_height - image_with_transparency.shape[0]) / 2) +1
position_x = round((desired_width - image_with_transparency.shape[1]) / 2) +1
print(f"Height of mask {mask.shape[0]}, Width of mask {mask.shape[1]}")
# Create a new canvas with an alpha channel
print(f"Desired height and width {desired_height, desired_width}")
#extended_canvas = numpy.ones((desired_height, desired_width, focusedImage.shape[2]), dtype=numpy.uint8)

# Copy the original image onto the new canvas at the calculated position
#extended_canvas[position_y:position_y + image_with_transparency.shape[0], position_x:position_x + image_with_transparency.shape[1], :] = mask

#print(f"Height of focus image {focusedImage.shape[0]}, Width of focus image {focusedImage.shape[1]}")
#print(f"Height of extended {extended_canvas.shape[0]}, Width of extended {extended_canvas.shape[1]}")
# Display the extended canvas with the image
#cv2.imshow('Extended Canvas', extended_canvas)

increments = 90
rotation_angle = 0  # You can change this angle as needed
while rotation_angle != 450:
    # Get the center of the image for rotation
    height, width = mask.shape[0], mask.shape[1]
    center = (width // 2, height // 2)
    #cv2.imshow("Focuse Image",focusedImage)
    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(mask, rotation_matrix, (width, height))

    # Perform bitwise AND operation with the rotated image and the second image
    result_image = (cv2.bitwise_and(focusedImage,rotated_image))
    cv2.imshow("Result Image",result_image)
    coverage_percentage =  numpy.sum(focusedImage != 255) / numpy.sum(result_image != 255) * 100
    print(f"Percentage coverate at angle {rotation_angle} is {coverage_percentage}")
    rotation_angle += increments
    
    #cv2.imshow("BITEWISE",result_image)
    #waitKey(100)







end_time = time.time()

# Calculate the runtime
runtime_seconds = end_time - start_time
runtime_milliseconds = runtime_seconds * 1000

print(f"Runtime: {runtime_milliseconds:.2f} milliseconds")

cv2.waitKey(0)
cv2.destroyAllWindows()
