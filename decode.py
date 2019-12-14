import numpy
import cv2

#                  0 = default device
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    cv2.imshow("WEGOLD", frame)

    if cv2.waitKey(16) == ord('q'):
        break

#webcam.release()
cv2.destroyAllWindows()
