# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
from collections import OrderedDict
import dlib
import cv2
import numpy as np

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("nose", (27, 35)),
    ("eyes", (36, 47)),
    ("jaw", (0, 16)),
    ("all", (0, 68))
])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

glass_img1 = cv2.imread('glass_image.jpg')
glass_img2 = cv2.imread('h.png')
glass_img3 = cv2.imread('dsd.png')

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

faceParts = FACIAL_LANDMARKS_IDXS.get("all")
glass_img = glass_img2
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        if shape.size == 0:
            print("CANNOT FIND FACE")
        else:
            if faceParts == FACIAL_LANDMARKS_IDXS.get("eyes"):
                # change the given value of 2.15 according to the size of the detected face
                glasses_width = 2.15*abs(shape[41][0]-shape[46][0])
                overlay_img = np.ones(frame.shape, np.uint8)*255
                h, w = glass_img.shape[:2]
                scaling_factor = glasses_width/w
                overlay_glasses = cv2.resize(glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                x = shape[46][0]
                y = shape[46][1]
                #   The x and y variables  below depend upon the size of the detected face.
                #x -= 0.26*overlay_glasses.shape[1]
                #y += 0.4*overlay_glasses.shape[0]
                #slice the height, width of the overlay image.
                h, w  = overlay_glasses.shape[:2]
                x -= 0.75 * w
                y -= h/2
                overlay_img[int(y):int(y+h),int(x):int(x+w)] = overlay_glasses

                #   Create a mask and generate it's inverse.
                gray_glasses = cv2.cvtColor(overlay_img,   cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(gray_glasses, 110,    255,    cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                temp = cv2.bitwise_and(frame,  frame,  mask=mask)
                temp2 = cv2.bitwise_and(overlay_img,    overlay_img,    mask=mask_inv)
                final_img = cv2.add(temp,   temp2)
            else:
                final_img = None

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        index = 0
        for (x, y) in shape:
                if index >= faceParts[0] and index <= faceParts[1]:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                index += 1
    
    # show the frame
    if faceParts == FACIAL_LANDMARKS_IDXS.get("eyes") and final_img is not None:
        cv2.imshow("Frame", final_img)
    else:
        cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    elif key == ord("m"):
        faceParts = FACIAL_LANDMARKS_IDXS.get("mouth")
        continue
    elif key == ord("e"):
        faceParts = FACIAL_LANDMARKS_IDXS.get("eyes")
        continue
    elif key == ord("j"):
        faceParts = FACIAL_LANDMARKS_IDXS.get("jaw")
        continue
    elif key == ord("a"):
        faceParts = FACIAL_LANDMARKS_IDXS.get("all")
        continue
    elif key == ord("n"):
        faceParts = FACIAL_LANDMARKS_IDXS.get("nose")
        continue
    elif key == ord("1"):
        glass_img = glass_img1
        continue
    elif key == ord("2"):
        glass_img = glass_img2
        continue
    elif key == ord("3"):
        glass_img = glass_img3
        continue

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
