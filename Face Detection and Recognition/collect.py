import cv2
import numpy as np

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data=[]

file_name=input('Enter the name of the person :')
while True:
    ret,frame=cap.read()
    
    if ret==False:
        continue
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5) #List of Tuples
    # 1.3-scaling parameter and 5- n nearest neighbours
    faces=sorted(faces,key= lambda f:f[2]*f[3])
    
    
    for face in faces[-1:]:
        x,y,w,h=face
        #Draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        offset=10
        #Extract the face
        face_section=frame[y-offset:y+h+offset,x-10:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        skip+=1
        
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
                    
    cv2.imshow('Video Capture',frame)
    cv2.imshow("Face",face_section)
    
    
    
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
#Flatten the largest face image(gray_scale) which is a list array and store it in a numpy array 
face_data=np.asarray(face_data)    
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save this data into file system 
np.save(file_name+'.npy',face_data)

        
cap.release()
cv2.destroyAllWindows()