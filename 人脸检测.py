import cv2 as cv

def face_detect():
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detecter = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    face = face_detecter.detectMultiScale(gray,1.01,5,0,(100,100),(1000,1000))
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=2)
    cv.imshow("result",img)

img=cv.imread("face1.png")
face_detect()
while True:
    if ord('q')==cv.waitKey(0):
        break

cv.destroyAllWindows()