import cv2
import os


def take_new_user_picture(id):
    # dir = name  # the name will be used to create a directory to store that person's images
    # parent_dir = "dataset"  # the directory to hold all images
    # path = os.path.join(parent_dir, dir)
    # os.mkdir(path)  # create the directory here

    # openCv to use the webcam,here the 1 referes to the source, and we are using iphone camera for
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press space to take a photo", 500, 300)

    img_counter = 0  # to label each image with a number

    while True:
        ret, frame = cam.read()  # ret is a return value when the camera is open
        if not ret:  # upon failure
            print("failed to grab frame")
            break
        cv2.imshow("press space to take a photo", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "dataset/" + id + "/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

