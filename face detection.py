import cv2
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)
image_path="faces1.jpg.png"
img=cv2.imread(image_path)
while True:
    text = " No Face detected"
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)
        text = "Face detected"
    print(text)
    cv2.imshow("FaceDetection",img)
    
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
