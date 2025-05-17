import os
import pandas as pd

root_folder= "E:\@Career\AI\Technology\Main Use\ML\College _Content\PCA_Image_Recognition\Data"

Data=[]

for label in os.listdir(root_folder):
    label_path=os.path.join(root_folder, label)
    if os.path.isdir(label_path):
        for filename in os.listdir(label_path):
            file_path=os.path.join(label_path, filename)
            if filename.lower().endswith('.pgm'):
                Data.append({'filepath' : file_path , 'label' : label})

df = pd.DataFrame(Data)


#Convert Path TO Matrix
# pip install opencv-python # in terminal
import cv2
def load_image_cv(path):
    img=cv2.imread(path , cv2.IMREAD_GRAYSCALE)
    return img

df['image_array'] = df['filepath'].apply(load_image_cv)

df=df.drop(["filepath"],axis=1)



#Convert Matrix TO Array
def matrix_to_vector(array):
    vector= array.ravel()
    print(len(vector))
    return vector

df["image_vector"]=df['image_array'].apply(matrix_to_vector)

df=df.drop(["image_array"],axis=1)


# Save_Data
print(df.head())
df.to_csv("Data/data_frame.csv", index=False)

