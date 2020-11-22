# This script serves to work on face recognition and landmark detection using
# OpenCV. Also calculates the occurlar lengths
# Runs dlib 68-point landmark detection
# %%
import cv2 as cv
import matplotlib.pyplot as plt
# https://morioh.com/p/d313e9b5c65f
# Read image from system
# %%
origin_image = cv.imread("F:\\FaceExprDecode\\F001\\T1\\0004.jpg")
grayscale_image = cv.cvtColor(origin_image, cv.COLOR_BGR2GRAY)

# Histogram normalization
hist_n = cv.equalizeHist(grayscale_image)
plt.imshow(hist_n)
# %%
face_cascade = cv.CascadeClassifier("C:\ProgramData\Anaconda3\envs\pytorches\Library\etc\haarcascades\haarcascade_frontalface_alt.xml")
detected_faces = face_cascade.detectMultiScale(grayscale_image)

# %%
face_landmark = cv.face.createFacemarkLBF()
face_landmark.loadModel('F:/lbfmodel.yaml.txt')
ok,landmarks = face_landmark.fit(grayscale_image,detected_faces)

# %%
for (column, row, width, height) in detected_faces:
    cv.rectangle(
        origin_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
plt.imshow(origin_image)
#cv.waitKey(0)
#cv.destroyAllWindows()
# %%
for landmark in landmarks[0][0]:
    cv.circle(origin_image,(landmark[0],landmark[1]),radius=2,color=(255, 0, 0),thickness = 2)
 
plt.imshow(origin_image)

#plt.axis("off")
#plt.imshow(grayscale_image)
# %%
