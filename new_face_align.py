import cv2
import numpy as np
import os
import math
import skimage
import matplotlib.pyplot as plt
# Many parts of the code derived from https://github.com/ZhiwenShao/PyTorch-JAANet/blob/master/dataset/face_transform.py
# Modified it to fit the 68 point face landmarks
#face_cascade = cv2.CascadeClassifier("C:\ProgramData\Anaconda3\envs\pytorches\Library\etc\haarcascades\haarcascade_frontalface_alt.xml")
#face_landmark = cv2.face.createFacemarkLBF()
#face_landmark.loadModel('F:/lbfmodel.yaml.txt')

def align_face_68pts(img, landmarks, box_enlarge, img_size):
    """
    Similarity Transformation, which performs in-plane rotation, uniform scaling and translation.
    Results output is 200x200x3 image and transformed landmarks. 
    """


    assert landmarks.shape == (
        68, 2), "You must pass a 68 point face landmark with x and y coordinates"
    lefteye_x = np.mean([landmarks[36, 0]+landmarks[37, 0]+landmarks[38, 0] +
                 landmarks[39, 0]+landmarks[40, 0]+landmarks[41, 0]])
    lefteye_y = np.mean([landmarks[36, 1]+landmarks[37, 1]+landmarks[38, 1] +
                 landmarks[39, 1]+landmarks[40, 1]+landmarks[41, 1]])
    righteye_x = np.mean([landmarks[42, 0]+landmarks[43, 0]+landmarks[44, 0] +
                  landmarks[45, 0]+landmarks[46, 0]+landmarks[47, 0]])
    righteye_y = np.mean([landmarks[42, 1]+landmarks[43, 1]+landmarks[44, 1] +
                  landmarks[45, 1]+landmarks[46, 1]+landmarks[47, 1]])

    change_x = righteye_x-lefteye_x
    change_y = righteye_y-lefteye_y

    l = np.sqrt(change_x**2+change_y**2)
    sin_val = change_y/l
    cos_val = change_x/l

    mat1 = np.mat([[cos_val, sin_val, 0], [-sin_val, cos_val, 0], [0, 0, 1]])

    mat2 = np.mat([[lefteye_x, lefteye_y, 1], [righteye_x, righteye_y, 1],
                           # This is nose tip
                           [landmarks[30, 0], landmarks[30, 1], 1],
                           # This is leftmost corner of mouth
                           [landmarks[48, 0], landmarks[48, 1], 1],
                           # This is rightmost corner of mouth
                           [landmarks[54, 0], landmarks[54, 1], 1]
                           ])

    mat2 = (mat1 * mat2.T).T

    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5

    if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))
    halfSize /= 4
    cx /= 4
    cy /= 4

    scale = (img_size - 1) / 2.0 / halfSize
    #scale = 0.25
    # Mat 3 is a scaling & Translation matrix
    mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
    mat = mat3 * mat1

    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    


    land_3d = np.ones((landmarks.shape[0], 3))
    land_3d[:, 0:2] = landmarks
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    print(new_land.shape)
    #new_land = np.reshape(new_land[:, 0:2], len(img_land))

    return (aligned_img, new_land)

def ocular_dist(landmarks):

    lefteye_x = np.mean([landmarks[36, 0]+landmarks[37, 0]+landmarks[38, 0] +
                 landmarks[39, 0]+landmarks[40, 0]+landmarks[41, 0]])
    lefteye_y = np.mean([landmarks[36, 1]+landmarks[37, 1]+landmarks[38, 1] +
                 landmarks[39, 1]+landmarks[40, 1]+landmarks[41, 1]])
    righteye_x = np.mean([landmarks[42, 0]+landmarks[43, 0]+landmarks[44, 0] +
                  landmarks[45, 0]+landmarks[46, 0]+landmarks[47, 0]])
    righteye_y = np.mean([landmarks[42, 1]+landmarks[43, 1]+landmarks[44, 1] +
                  landmarks[45, 1]+landmarks[46, 1]+landmarks[47, 1]])

    ocular_distance = (lefteye_x - righteye_x) ** 2 + (lefteye_y - righteye_y) ** 2

    return(ocular_distance)

def preprocess_image(img, Enlarge_fac = 2.9, img_size = 200, face_landmark = None, face_cascade = None): 

    ENLARGE_FACTOR = Enlarge_fac
    IMG_SIZE = img_size

    img = np.array(img)
    #Debug here
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(grayscale_image)

    x = detected_faces[0][0]
    y = detected_faces[0][1]
    w = detected_faces[0][2]
    h = detected_faces[0][3]

    multiplier_h_down = np.max([0,int(y-0.5*h)])
    multiplier_h_up = np.min([img.shape[0],int(y + 1.5*h)])
    multiplier_w_down = np.max([0,int(x-0.5*w)])
    multiplier_w_up =  np.min([img.shape[1],int(x + 1.5*w)])
    crop_img = img[multiplier_h_down:multiplier_h_up,multiplier_w_down:multiplier_w_up,:]
    grayscale_image_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    new_detected_frame = np.array([[int(x-multiplier_w_down),int(y-multiplier_h_down),w,h]])

    ok, landmarks = face_landmark.fit(grayscale_image_crop, new_detected_frame)
    new_img, new_landmarks = align_face_68pts(crop_img, landmarks[0][0], ENLARGE_FACTOR, IMG_SIZE)
    new_ocular_dist = ocular_dist(new_landmarks)
    return(new_img, new_landmarks, new_ocular_dist)




#img = cv2.imread(r'C:\Users\Yaqian\Downloads\image.png',0)

# face_cascade = cv2.CascadeClassifier("C:\ProgramData\Anaconda3\envs\pytorches\Library\etc\haarcascades\haarcascade_frontalface_alt.xml")
# face_landmark = cv2.face.createFacemarkLBF()
# face_landmark.loadModel('F:/lbfmodel.yaml.txt')

# #img_file = 'F:\\FaceExprDecode\\F001\\T1\\0000.jpg'
# #img_file = 'C:\\Users\\Yaqian\\Downloads\\image3.png'
# img_file = 'C:\\Users\\Yaqian\\Downloads\\JinHyunCheong.jpg'

# image = cv2.imread(img_file)
# (new_img, new_landmarks, new_ocular_dist) = preprocess_image(image, Enlarge_fac = 1.5, face_landmark = face_landmark, 
#                                                             face_cascade = face_cascade)
# #print(new_img.shape)
# plt.imshow(new_img)
# plt.scatter(new_landmarks[:,0],new_landmarks[:,1], s=1)
# plt.show()
