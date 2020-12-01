# Run viola johnes face detector to remove pictures that do not have a face

# %%
import pandas as pd
import glob
import cv2
from PIL import Image
import numpy as np

gen_path = "F:\\FaceExprDecode\\"
subj_names = ["F001","M001"]
task_names = [["T1","T6","T7","T8"],["T1","T6","T7","T8"]]
face_cascade = cv2.CascadeClassifier("C:\ProgramData\Anaconda3\envs\pytorches\Library\etc\haarcascades\haarcascade_frontalface_alt.xml")

def _construct_imag_csv(gen_path, subj_names, task_names, face_cascade = face_cascade):
    """
    Concatenate general directory csv file from images in the folder, and combining
    it with Action Unit labels.
    Params:
        gen_path: a general path for all files
        subj_names: subject names
        task_names: task names for the subject
    """
    pic_subj = []
    pic_task = []
    pic_num = []
    result_arr = None
    for i, subj in enumerate(subj_names):
        
        for j, task in enumerate(task_names[i]):
            #print(gen_path+subj+"\\"+task+"\\"+"*.jpg")
            curr_jpgs = glob.glob(gen_path+subj+"\\"+task+"\\"+"*.jpg")
            valid_jpgs = []
            for img_file in curr_jpgs:
                img = Image.open(img_file)
                img = np.array(img)
                grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detected_faces = face_cascade.detectMultiScale(grayscale_image)
                if len(detected_faces) > 0:
                    valid_jpgs.append(img_file)
            #face_landmark = cv2.face.createFacemarkLBF()
            #face_landmark.loadModel('F:/lbfmodel.yaml.txt')

            pic_subj += [subj]*len(valid_jpgs)
            pic_task += [task]*len(valid_jpgs)
            pic_num += [i.split("\\")[-1].split(".")[0] for i in valid_jpgs]
            
            temp_df1 = pd.DataFrame(list(zip(pic_subj, pic_task, pic_num)), 
               columns =['Subject', 'Task', 'Number'])
            
            temp_AU1 = pd.read_csv(gen_path + subj+"_"+"label\\" + subj + "_" + task + ".csv")
            temp_AU1.rename(columns={'1':'Number','1.1':'1'}, inplace=True)
            temp_AU1 = temp_AU1.astype({'Number':'str'})
            result = pd.merge(temp_df1, temp_AU1, how = 'inner', on = "Number")
            
            if result_arr is None:
                result_arr = result
            else:
                result_arr = pd.concat([result_arr, result], axis=0)

    pic_subj = [item for sublist in pic_subj for item in sublist]
    pic_task = [item for sublist in pic_task for item in sublist]
    pic_num = [item for sublist in pic_num for item in sublist]
    
    return(result_arr)

        


# %%
random01 = _construct_imag_csv(gen_path, subj_names, task_names, face_cascade = face_cascade)
# %%
