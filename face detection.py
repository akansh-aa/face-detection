import cv2
# create a cascade classifier
face_cascasde=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
img=cv2.imread("pic.jpg")
# Reading the image at grayscale
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#search the coordinates oof image
faces=face_cascasde.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)
print(type(faces))
print(faces)

for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
resized=cv2.resize(img,(int(img.shape[1]/2),(int(img.shape[0]))))
cv2.imshow("IMAGE",img)
cv2.waitKey(0)
cv2.destroyAllWindows()