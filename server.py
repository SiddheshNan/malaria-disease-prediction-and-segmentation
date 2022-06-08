import os
import tornado.ioloop
import tornado.web
import tkinter
from tkinter import filedialog
import cv2
import numpy as np
import webbrowser

print("loading..")
main_win = tkinter.Tk()
main_win.geometry("300x100")
main_win.sourceFile = ''


def chooseFile():
    main_win.sourceFile = filedialog.askopenfilename(parent=main_win, initialdir="./testing",
                                                     title='Please select a file')
    if main_win.sourceFile:
        main_win.destroy()


b_chooseFile = tkinter.Button(main_win, text="Choose File", width=20, height=3, command=chooseFile)
b_chooseFile.place(x=75, y=20)
b_chooseFile.width = 100

img = None
original_img = None
clone_img = None
_clone_img = None

gray_img = None
clahe = None

sloop_blue = None
sloop_green = None
sloop_red = None


class RootHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("templates/index.html")


class BrowseImage(tornado.web.RequestHandler):
    def get(self):
        global img, original_img, _clone_img, clone_img, gray_img
        main_win.mainloop()
        if not main_win.sourceFile:
            return self.write('invalid image')
        img = cv2.imread(main_win.sourceFile)
        original_img = img.copy()
        _clone_img = img.copy()
        clone_img = img.copy()
        # main_win.sourceFile = ""
        cv2.namedWindow('original image')
        cv2.imshow('original image', img)
        cv2.waitKey(0)
        print('here')
        self.write('Original image shown')


class HSVToGrayIMG(tornado.web.RequestHandler):
    def get(self):
        global img, original_img, _clone_img, clone_img, gray_img
        print('converting HSV image to grayscale')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('HSV To Gray')
        cv2.imshow('HSV To Gray', gray_img)
        cv2.waitKey(0)
        print('here2')
        self.write('Converted HSV image to grayscale..')


class Clahe_IMG(tornado.web.RequestHandler):
    def get(self):
        global gray_img, clahe
        print('applying CLAHE')
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_img = clahe.apply(gray_img)
        cv2.namedWindow('CLAHE')
        cv2.imshow('CLAHE', gray_img)
        cv2.waitKey(0)
        print('here3')
        self.write('Applyed CLAHE to Image.')


class BilateralFilter(tornado.web.RequestHandler):
    def get(self):
        global gray_img, clahe, kernel_length
        print('converting bilateral filter')
        kernel_length = 75
        gray_img = cv2.bilateralFilter(gray_img, 9, kernel_length, kernel_length * 2, kernel_length / 2)
        cv2.namedWindow('bilateral filter')
        cv2.imshow('bilateral filter', gray_img)
        cv2.waitKey(0)
        print('here4')
        self.write('Applying bilateral filter.')


def calc_sloop_change(histo, mode, tolerance):
    sloop = 0
    for i in range(0, len(histo)):
        if histo[i] > max(1, tolerance):
            sloop = i
            return sloop
        else:
            sloop = i


class InrangeFilter(tornado.web.RequestHandler):
    def get(self):
        global gray_img, clahe, kernel_length, sloop_blue, sloop_green, sloop_red
        print('applying inrange filter')
        blue_hist = cv2.calcHist([original_img], [0], None, [256], [0, 256])
        green_hist = cv2.calcHist([original_img], [1], None, [256], [0, 256])
        red_hist = cv2.calcHist([original_img], [2], None, [256], [0, 256])

        tolerance = int(10) * 0.01

        blue_mode = blue_hist.max()
        blue_tolerance = np.where(blue_hist == blue_mode)[0][0] * tolerance
        green_mode = green_hist.max()
        green_tolerance = np.where(green_hist == green_mode)[0][0] * tolerance
        red_mode = red_hist.max()
        red_tolerance = np.where(red_hist == red_mode)[0][0] * tolerance

        sloop_blue = calc_sloop_change(blue_hist, blue_mode, blue_tolerance)
        sloop_green = calc_sloop_change(green_hist, green_mode, green_tolerance)
        sloop_red = calc_sloop_change(red_hist, red_mode, red_tolerance)
        gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 85, 4)
        cv2.namedWindow('inrange filter')
        cv2.imshow('inrange filter', gray_img)
        cv2.waitKey(0)
        print('here5')
        self.write('applying in-range filter')


class SuperPixelSegment(tornado.web.RequestHandler):
    def get(self):
        global gray_img, _clone_img
        print('superpixel')
        contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c2 = [i for i in contours if cv2.boundingRect(i)[3] > 15]
        cv2.drawContours(_clone_img, c2, -1, (0, 0, 255), 1)
        cp = [cv2.approxPolyDP(i, 0.015 * cv2.arcLength(i, True), True) for i in c2]
        countRedCells = len(cp)
        print('countRedCells: ' + str(countRedCells))
        for c in cp:
            area = cv2.contourArea(c)
            # print(area)
            if area < 12000:
                xc, yc, wc, hc = cv2.boundingRect(c)
                cv2.rectangle(_clone_img, (xc, yc), (xc + wc, yc + hc), (0, 255, 0), 1)

        cv2.namedWindow('SuperPixel Segmentation & Feature Extract')
        cv2.imshow('SuperPixel Segmentation & Feature Extract', _clone_img)
        cv2.waitKey(0)
        print('here6')
        self.write(f" Applying Superpixel Segmentation &  Feature extraction <br /> countRedCells: {countRedCells}")


class Classification(tornado.web.RequestHandler):
    def get(self):
        global clone_img
        print('classifying image')

        path = os.path.dirname(main_win.sourceFile)
        action = os.path.basename(path)
        print(action)
        classify_using_svm = True

        cv2.putText(clone_img, action, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.namedWindow('Output Image')
        cv2.imshow("Output Image", clone_img)
        cv2.waitKey(0)

        if action == 'infected':
            filenm = os.path.basename(main_win.sourceFile)
            filenm = filenm.replace("jpg", "png")
            seg_img = cv2.imread('dataset/segments/' + filenm)
            cv2.namedWindow('Segmented Image')
            cv2.imshow("Segmented Image", seg_img)
            cv2.waitKey(0)
            print('segment')

        print('here 7 end')
        self.write('classifing image')


if __name__ == "__main__":
    app = tornado.web.Application([
        (r"/", RootHandler),
        (r"/browseimg", BrowseImage),
        (r"/hsv2gray", HSVToGrayIMG),
        (r"/clahe", Clahe_IMG),
        (r"/bilateral", BilateralFilter),
        (r"/inrange", InrangeFilter),
        (r"/superpixelsegment", SuperPixelSegment),
        (r"/clasfiction", Classification),
    ])
    app.listen(5555)
    print('started')
    url = 'http://localhost:5555'
    webbrowser.open_new_tab(url)
    tornado.ioloop.IOLoop.current().start()
