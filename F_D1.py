import cv2
from random import randrange

#load some pre-trained data on face frontals from opencv(haar cascade algo)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose and image to detect face
#img=cv2.imread('multiple.jpg')


#to capture web cam
webcam=cv2.VideoCapture(0)

#Itereate forever over frames
while True:

	#read the current frame
	successful_frame_read, frame=webcam.read()

	#must convert to grayscale
	grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


	#detect faces
	face_coordinates= trained_face_data.detectMultiScale(grayscaled_img,minSize=(60,60))
	

	for (x,y,w,h) in face_coordinates:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),4)	

	cv2.imshow('Face detected',frame)
	key=cv2.waitKey(1)

	#ascii value of Q and q
	if key==81 or key==113:
		break

#webcam release
webcam.release()
'''
#must convert to grayscale
grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#detect faces
face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)
 
#draw rect around the faces
for (x,y,w,h) in face_coordinates:
	cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

#print(face_coordinates)


#display the image
cv2.imshow('Face detected',img)
cv2.waitKey()

print('Code Completed')
'''

