#! /usr/bin/python
import numpy
import numpy as np
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
from datetime import datetime
import pandas as pd
from datetime import date
from datetime import datetime

#### Add Attendance of a specific user
datetoday = date.today().strftime("%m_%d_%y")
def run_facial_recognition_system():
    # Initialize 'currentName' to trigger only when a new person is identified.
    currentName = "Unknown"

    # Determine faces from encodings.pickle file model created from facial_recognition_model.py
    encodingsP = "encodings.pickle"

    # load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(encodingsP, "rb").read())

    # initialize the video stream and allow the camera sensor to warm up
    # Set the source to the following
    # src = 0 : for the build in single webcam, could be your laptop webcam
    # src = 1 : using continuity feature on the new macOS ventura, open iphone camera for better quality
    # src = 2 : USB webcam attached to my laptop

    vs = VideoStream(src=0, framerate=10).start()
    # vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    # start the FPS counter
    fps = FPS().start()

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500, height=500)

        # Detect the face boxes
        boxes = face_recognition.face_locations(frame)

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []

        # loop over the facial embeddings
        tolerance = 0.45
        for encoding in encodings:
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=tolerance)
            name = "Unknown"  # if face is not recognized, then print Unknown
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            # best_match_index = np.argmin(face_distances)
            # print("best match: ", best_match_index)

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIndexes = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # calculate the accuracy in oercentages
                # face_distances = face_recognition.face_distance(data["encodings"], encoding)
                max_distance = min(face_distances)
                face_match_percentage = (1 - max_distance) * 100

                # loop over the matched indexes and maintain a count for
                # each recognized face
                for i in matchedIndexes:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

                # If someone in the dataset is identified, print their name on the screen
                if currentName != name:
                    currentName = name
                    # print(face_accuracy)
                    print(currentName)
                    print("This person is identified as {} with {} accuracy".format(currentName, np.round(face_match_percentage, 4)))

            # update the list of names
            names.append(name)
            # if name != "Unknown":

            # #########################
            # if name == "Unknown":
            #     print("Alert!")
            #     break
            ##############################

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            if name != "Unknown":
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (255, 0, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            .8, (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 0, 255), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            .8, (0, 0, 255), 2)

        # display the image to our screen
        cv2.imshow("Facial Recognition is Running", frame)
        key = cv2.waitKey(1) & 0xFF

        # quit when 'escape' key is pressed
        if key % 256 == 27:
            print("Escape key pressed, shutting down")
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
