import numpy
import tkinter
from PIL import Image, ImageTk
import cv2
import test as model

window = tkinter.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

webcam = cv2.VideoCapture(0)

webcamFrame = tkinter.Frame(window, width=screen_width, height=screen_height)
webcamLabel = tkinter.Label(window)
webcamLabel.pack()
aslCharText = tkinter.Text(window, height=2, width=30, bd = 0)
aslCharText.pack()

frameCount = 0
previousCharacter = 'a'
def update():
    global frameCount, previousCharacter
    _, frame = webcam.read()

    imgtk = ImageTk.PhotoImage(
            image=Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)))
    webcamLabel.imgtk = imgtk 
    webcamLabel.configure(image=imgtk)

    if frameCount >= 15:
        frameCount = 0
        cv2.imwrite('asl.jpg', frame)
        character = model.aslToChar('asl.jpg')
        if character == 'del':
            aslCharText.delete("end-2c")
        elif character != "nothing" and character != previousCharacter:
            aslCharText.insert(tkinter.END, character)
        previousCharacter = character


    webcamLabel.after(16, update)
    frameCount += 1

update()
window.mainloop()
