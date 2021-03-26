### Importing libraries
import cv2
import numpy as np
import face_recognition
import os                                        ### for getting a list of names of files from another directory
from datetime import datetime


path = 'ImagesAttendance'
images = []                                      ### stores image names in .jpg
classNames = []                                  ### image names only
myList = os.listdir(path)
print(myList)

### Getting class names
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])   ### to get without '.jpg', for putting text on recognised face
#print(images[0].shape)                           ### pixel values in array
print(classNames)


def attendance(name):
    with open('attendance.csv', 'r+') as f:
       mydatalist= f.readlines()
       namelist= []
       for line in mydatalist:
           entry= line.split(',')
           namelist.append(entry[0])
       if name not in namelist:                                       ### only once marked
           time= datetime.now()
           timeString=time.strftime("%H:%M:%S")
           f.writelines(f'\n{name},{timeString}')
#attendance('Elon')


def find_encoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                    ### images need to be in RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = find_encoding(images)                               ### trained images encoding
#print(len(encodeListKnown))
print("Encoding Finished")

video = cv2.VideoCapture(0)
video.set(3,640)  #id number 3 ---> width
video.set(4,480)   #id 4 -->height
video.set(10,100)

while True:
    success, img = video.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)                        ### speeds up process by reducing 1/4th of size
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesPresent = face_recognition.face_locations(imgSmall)                   ### finds multiple faces location
    encodesPresent = face_recognition.face_encodings(imgSmall, facesPresent)


    ### now compare encoding of webcam faces to known encodings
    for encodeFace, faceLoc in zip(encodesPresent, facesPresent):               ### each face pointed
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)   ### comparing one encoding to an list of known encodings
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)   ### gives a list of three distances
        matchIndex = np.argmin(faceDis)                                         ### minimum distance
        if matches[matchIndex]:                                                 ### if comparison is true
            name = classNames[matchIndex].upper()
            print(name)
            print(faceDis)

            ### Bounding Box fixed code by author
            y1, x2, y2, x1 = faceLoc                                            ### locations of matched face
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4                     ### because we resized the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break