from tqdm import tqdm
import face_recognition
import glob
import csv
import numpy as np

def generate_training_data(folder):
    r=0
    print("Generating encodings for db images..")
    image_encodings=[]
    with tqdm(total=len(glob.glob(folder+"/*.jpg"))) as pbar:
        for img in glob.glob(folder+"/*.jpg"):
            enc=[]
            img_name=img[3:]
            known_img=face_recognition.load_image_file(img)
            try:
                en=face_recognition.face_encodings(known_img)[0]
            except:
                print("can't generate encodings for "+img_name+", give another image")
                pass
            #en=np.array(en)
            for e in en:
                enc.append(e)
            image_encodings.append([img_name,enc])
            pbar.update(1)
    return image_encodings

encodings=generate_training_data("db")
#print(encodings[0])

csvfile = "Encodings/encodings.csv"

i=0
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    if(i==0):
        writer.writerow(["Img_Name","Encoding"])
    i+=1
    writer.writerows(encodings)

print("Encodings updated for all images")

