import cv2
import face_recognition

imgElon = face_recognition.load_image_file('basicimages/elon-musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('basicimages/bill-gates-2.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

### Train image
faceLoc = face_recognition.face_locations(imgElon)[0]          ### single image passed
encodeElon = face_recognition.face_encodings(imgElon)[0]
#print(faceLoc)  ##top right bottom and left
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

### test image
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

### comparing both images
results = face_recognition.compare_faces([encodeElon],encodeTest)
### lower distance more similar the images
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
### putting text in image
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(faceLocTest[1],faceLocTest[3]),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),1)

print(results)  ## gives True if both are same
print(faceDis)  ## lower distance --> similarity

cv2.imshow("Test Image",imgTest)
cv2.imshow("Train Image",imgElon)

cv2.waitKey(0)