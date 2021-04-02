#keywords = pdi, opencv, numberplate, webservice, api
#OpenCV is an open source programming library designed to make computer vision more accessible.
#Numpy is a library highly optimized for numerical operations

import cv2
# universally unique identifier
import uuid
import numpy as np

##--*--##
#Cascading method is not the most accurate, but it is fast (works under circumstances).
nPlateCascade = cv2.CascadeClassifier("russianCar/haarcascade_russian_plate_number.xml")
minArea = 200
color = (255, 0, 255)
count = 0
# Read image with opencv
img = cv2.imread("russianCar/car2.jpg")
# Get image size
height, width = img.shape[:2]
# Scale image
img = cv2.resize(img, (800, int((height * 800) / width)))
# Converting to Gray Scale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Noise removal with iterative bilateral filter and canny(removes noise while preserving edges).
imgGray = cv2.bilateralFilter(imgGray, 11, 17, 17)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
imgCanny = cv2.Canny(imgBlur, 150, 200)

numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)
for (x, y, w, h) in numberPlates:
    area = w * h
    if area > minArea:
        # Put text license plate number to image
        cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        # Draw rectangle around license plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        imgCrop = imgGray[y:y + h, x:x + w]
        imgRoi = cv2.Canny(imgCrop, 150, 200)
        # Region of interest
        cv2.imshow("ROI", imgRoi)
        cv2.imshow("ROI2", imgCrop)
        
    cv2.imshow("Russian Car", img)

    # Crop those contours and store it in Cropped Images folder
    if cv2.waitKey(0) & 0xFF == ord('s'):
        # A unique session ID every time
        cv2.imwrite("russianCar/licensePlateSaved/" + uuid.uuid4().hex + ".jpg", imgRoi)
        cv2.imwrite("russianCar/licensePlateSaved/" + uuid.uuid4().hex + ".jpg", imgCrop)
        cv2.rectangle(img, (0, 200), (1280, 300), (0, 255, 255), cv2.FILLED)
        cv2.putText(img, "Saved Plate", (200, 265), cv2.FONT_HERSHEY_DUPLEX,
                    2, (0, 0, 255), 5)
        cv2.imshow("Russian Car", img)
        cv2.waitKey(500)
        count += 1
        cv2.destroyAllWindows()
