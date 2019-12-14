import numpy
import tkinter
from PIL import Image, ImageTk
import cv2
import test as model

window = tkinter.Tk()
webcam = cv2.VideoCapture(0)

webcamFrame = tkinter.Frame(window, width=1280, height=720)
lmain = tkinter.Label(window)
lmain.pack()

frameCount = 0
def update():
    global frameCount
    _, frame = webcam.read()

    imgtk = ImageTk.PhotoImage(
            image=Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)))
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)

    if frameCount >= 15:
        frameCount = 0
        cv2.imwrite('asl.jpg', frame)
        print(model.aslToChar('asl.jpg'), end="", flush=True)

    lmain.after(16, update)
    frameCount += 1

update()
window.mainloop()
