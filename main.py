import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# Upper and lower HSV bounds for the color green
lower_bound = np.array([50, 20, 20])   
upper_bound = np.array([100, 255, 255])


while(True):
    ret, img = cap.read()

    # Blur the source image and convert it to the HSV colorspace
    blurred_input = cv2.GaussianBlur(img, (11, 11), 0)
    hsv_input = cv2.cvtColor(blurred_input, cv2.COLOR_BGR2HSV)

    # Mask out the green of the cone in the image and break it down to cleaner shapes
    mask_input = cv2.inRange(hsv_input, lower_bound, upper_bound)
    mask_input = cv2.inRange(hsv_input, lower_bound, upper_bound)
    mask_input = cv2.erode(mask_input, None, iterations=2)
    mask_input = cv2.dilate(mask_input, None, iterations=2)

    # Run a countour search for the countours in the image
    cone_contours, _ = cv2.findContours(mask_input.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if(len(cone_contours) > 0):
            # Find the max value of the countours array sorting by area
            best_cone = max(cone_contours, key=cv2.contourArea)

            # Find the best fit rectangle in the largest contour
            rect = cv2.minAreaRect(best_cone)
            x = rect[0][0]
            y = rect[0][1]

            # Convert the rectangle into a series of vertex points to draw on screen
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0,0,255), 2)
    

    cv2.imshow('Found Cones', img)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break