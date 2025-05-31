import cv2 as cv
def face_detect_video(img):
    gray = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
    face_detect = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    face = face_detect.detectMultiScale(gray)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow("result",img)

cap = cv.VideoCapture(0)
cap.read()

while True:
    flag,frame =cap.read()
    if not flag:
        break

    face_detect_video(frame)
    if ord("q") == cv.waitKey(0):
        break

cv.destroyAllWindows()
cap.release()
