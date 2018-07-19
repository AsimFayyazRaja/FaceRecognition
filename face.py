from tqdm import tqdm
import face_recognition
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import ast
from scipy import misc as cv2


def get_encodings():
    df=pd.read_csv('Encodings/encodings.csv')       #getting file
    print("Loading all encodings..")
    with tqdm(total=len(list(df.iterrows()))) as prbar:
        encodings=[]
        images=[]
        for index, row in df.iterrows():
            r=[]
            en=df.loc[index,'Encoding']
            m=[]
            en=ast.literal_eval(en)
            en=np.array(en,dtype='<U1555')
            for e in en:
                m.append(float(e))
            encodings.append(m)
            img=df.loc[index,'Img_Name']
            images.append(img)
            prbar.update(1)
    return images,encodings



def generate_training_data(folder):
    r=0
    print("Getting images from db..")
    known_image=[]
    with tqdm(total=len(glob.glob(folder+"/*.jpg"))) as pbar:
        for img in glob.glob(folder+"/*.jpg"):
            known_image.append(face_recognition.load_image_file(img))
            pbar.update(1)
    return known_image

def generate_image(folder):
    r=0
    print("Getting images that matched..")
    with tqdm(total=len(glob.glob(folder))) as pbar:
        for img in glob.glob(folder):
            n= cv2.imread(img)
            pbar.update(1)
    return n


images,encodings=get_encodings()
print(len(images))
print(len(encodings))
i=0
matched_imgs=[]
unknown_image = face_recognition.load_image_file("testImage/preeti.jpg")    #test image name here
print("matching with encodings..")
with tqdm(total=len(encodings)) as pbar:
    for enc in encodings:
        known_encoding = enc
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces([enc], unknown_encoding)
        if(results[0]==True):
            matched_imgs.append(images[i])
        i+=1
        pbar.update(1)

for img in matched_imgs:
    p=generate_image("db/"+img)
    plt.imshow(p)
    plt.show()
