import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(
    'haarcascade_eye.xml')

# read both the images of the face and the glasses
image = cv2.imread('face.jpg')
glass_img = cv2.imread('glass_image.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

centers = []
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# iterating over the face detected
for (x, y, w, h) in faces:

    # create two Regions of Interest.
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # Store the cordinates of eyes in the image to the 'center' array
    for (ex, ey, ew, eh) in eyes:
        centers.append((x+int(ex+0.5*ew), y+int(ey+0.5*eh)))

if len(centers) > 0:

    # change the given value of 2.15 according to the size of the detected face
    print("tocke ociju: ", centers[1],centers[0])
    print("sirina naocala: ", glass_img.shape)
    glasses_width = 3*abs(centers[1][0]-centers[0][0])
    overlay_img = np.ones(image.shape, np.uint8)*255
    h, w = glass_img.shape[:2]
    scaling_factor = glasses_width/w
    overlay_glasses = cv2.resize(
        glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]

    #   The x and y variables  below depend upon the size of the detected face.
    x -= 0.26*overlay_glasses.shape[1]
    y += 0.85*overlay_glasses.shape[0]
    print("y: ", y,"\nx: ", x)
    print("h: ", h,"\nw: ", w)
    # slice the height, width of the overlay image.
    print("overlay glasses shape: ", overlay_glasses.shape)
    print("overlay image shape: ", overlay_img.shape)
    #overlay_img[int(y):int(y+h),    int(x):int(x+w)] = overlay_glasses

    #   Create a mask and generate it's inverse.
    gray_glasses = cv2.cvtColor(overlay_img,   cv2.COLOR_BGR2GRAY)
    ret,    mask = cv2.threshold(
        gray_glasses, 110,    255,    cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    temp = cv2.bitwise_and(image,  image,  mask=mask)
    temp2 = cv2.bitwise_and(overlay_img,    overlay_img,    mask=mask_inv)
    final_img = cv2.add(temp,   temp2)

    cv2.line(image, (centers[1][1], centers[1][0]), (centers[0][1],centers[0][0]), (0,255,0), 3)
    print("final image shape: ", final_img.shape)
    cv2.imshow('Lets wear Glasses', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
