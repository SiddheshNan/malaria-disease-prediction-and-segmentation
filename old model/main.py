import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import tkinter
from tkinter import filedialog
import cv2


main_win = tkinter.Tk()
main_win.geometry("300x100")

lables = open("lables.txt", "r").read()
actions = lables.split("\n")


main_win.sourceFile = ''


def chooseFile():
    main_win.sourceFile = filedialog.askopenfilename(
        parent=main_win, initialdir="./", title='Please select a File')
    if main_win.sourceFile:
        main_win.destroy()


b_chooseFile = tkinter.Button(
    main_win, text="Choose File", width=20, height=3, command=chooseFile)
b_chooseFile.place(x=75, y=20)
b_chooseFile.width = 100

main_win.mainloop()

if not main_win.sourceFile:
    print("invalid filepath")
    exit(1)

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('malaria.model', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open(main_win.sourceFile)

cv2_img = cv2.imread(main_win.sourceFile)

size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

image_array = np.asarray(image)

# cv2.imshow("Original Image", cv2_img)

cv2_resized = cv2.resize(cv2_img, size)
cv2_resized_bw = cv2.cvtColor(cv2_resized, cv2.COLOR_BGR2GRAY)

cv2.imshow("Resized Image", cv2_resized)
cv2.imshow("Resized Grayscale Image", cv2_resized_bw)


normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_image_array

prediction = model.predict(data)
prediction_new = prediction[0].tolist()

max_accuracy_action = max(prediction_new)
detected_action_index = prediction_new.index(max_accuracy_action)

print("Detected: " + actions[detected_action_index],
      "| Accuracy of Neural Network: " + str(max_accuracy_action))


detected_disease = actions[detected_action_index]

cv2.putText(cv2_img, detected_disease, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imshow("Output Image", cv2_img)


cv2.waitKey(10000)
