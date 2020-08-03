import cv2
import numpy as np
import os

#KNN CODE
def distance(v1,v2):
    return np.sqrt(sum((v1-v2)**2))   #Euclidean Distance 


def knn(train,test,k=5):
    dist=[]

    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        
        d=distance(test,ix)
        dist.append([d,iy])
        
    #Sort based on distance to get top k
    dk=sorted(dist,key=lambda x:x[0])[:k]
        
    #Retrieve only the labels
    labels=np.array(dk)[:,-1]
        
    #Get frequencies of each label
    output=np.unique(labels,return_counts=True)
        
    #Find max frequency and corresponding label
    index=np.argmax(output[1])
    return output[0][index]

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0;
face_data=[]
labels=[]
class_id=0
names={}

#Load the data
for fx in os.listdir():
    if fx.endswith('.npy'):
        print(fx)
        names[class_id]=fx[:-4]
        data_item=np.load(fx)
        print(data_item.shape)

        face_data.append(data_item)
        #Create labels
        target=class_id*np.ones((data_item.shape[0],))
        #print(target.shape)
        class_id+=1
        labels.append(target)
        #print(labels)
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))   

print(face_dataset.shape)
print(face_labels.shape)

trainset=np.concatenate((face_dataset,face_labels),axis=1) #axis=1 appends a column
print(trainset.shape)

while True:
    ret,frame=cap.read()
    
    if ret==False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5) 
    
    for face in faces:
        x,y,w,h=face
        
         #Get the face ROI
        offset=10
        #Extract the face
        face_section = frame[y-offset:y+h+offset,x-10:x+w+offset]

        face_section = cv2.resize(face_section,(100,100))
        
        out=knn(trainset,face_section.flatten())

        pred_name=names[int(out)]
        
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        
    cv2.imshow("Faces",frame)
    
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

