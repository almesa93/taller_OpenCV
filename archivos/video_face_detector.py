import cv2 as cv

haar_cascade = cv.CascadeClassifier('./Section #3 - Faces/haar_face.xml')
# capture = cv.VideoCapture('./Resources/Videos/alex_2.mp4')
capture = cv.VideoCapture(1)

while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=20)

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)

    if isTrue:
        cv.imshow('Video alex', frame)
        if cv.waitKey(20) & 0xFF==ord('q'):
            break
    
    else:
        break

capture.release()
cv.destroyAllWindows()