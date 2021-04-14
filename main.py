from PIL import Image, ImageDraw, ImageFont
from random import randint, random
from string import ascii_uppercase, ascii_lowercase, digits
import numpy as np
import cv2
import imutils
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import os
from skimage.util import random_noise


# Reference: https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
# For the given path, get the List of all files in the directory tree


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


os.system("mkdir -p data")
all_char = digits+ascii_uppercase
font_paths = []
for c in getListOfFiles("./fonts/"):
    if c[len(c)-4:] == ".ttf":
        font_paths.append(c)
myfonts = [ImageFont.truetype(font=i, size=90) for i in font_paths]

f_ind = 1
for char in all_char[:]:
    freq = 0
    for ind in range(1000):
        freq += 1
        font_color = (randint(240, 255), randint(240, 255), randint(240, 255))
        bg_color = (randint(0, 150), randint(0, 150), randint(0, 150))

        out = Image.new("RGB", (400, 400), font_color)
        font = myfonts[randint(0, len(myfonts)-1)]
        # print(font.getname()[0])
        d = ImageDraw.Draw(out)
        d.text((75, 40), char, font=font, fill=bg_color)
        out = out.rotate(20*random()-10)
        out.save("_.png")
        img = cv2.imread('_.png')

        # Read the image
        rows, cols = img.shape[:2]

        # Define the 3 pairs of corresponding points
        input_pts = np.float32(
            [
                [0, 0],
                [10, 10],
                [cols, rows]
            ]
        )

        output_pts = np.float32(
            [
                [0, 0],
                [10+(-2), 10+(-2)],
                # [10+(4*random()-2), 10+(4*random()-2)],
                [cols, rows]
            ]
        )

        # Calculate the transformation matrix using cv2.getAffineTransform()
        M = cv2.getAffineTransform(input_pts, output_pts)

        # Apply the affine transformation using cv2.warpAffine()
        dst = cv2.warpAffine(img, M, (cols, rows))


        img = img[30:170, 30:170]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        cv2.bitwise_not(thresh, thresh)
        cnts = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        c = max(cnts, key=cv2.contourArea)
        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])
        distance = np.sqrt((right[0] - left[0])**2 + (right[1] - left[1])**2)
        x, y, w, h = cv2.boundingRect(c)
        centx = np.sqrt(((right[0] + left[0])**2)/4)
        centy = np.sqrt(((right[1] + left[1])**2)/4)
        # print(centx, centy)
        # print(x, y)
        # print(w, h)

        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropped = img[y:y+h, x:x+w]
        # plt.imshow(cropped), plt.show()


        # print(h, w)
        b = 5
        resized = cv2.resize(cropped, (50, 50))
        padded = cv2.copyMakeBorder(
            resized,
            b, b, b, b,
            cv2.BORDER_CONSTANT, value=font_color)

        blur_val = randint(1, 4)
        blur = cv2.blur(padded, (blur_val, blur_val))

        noise_img = random_noise(blur, mode='s&p', amount=random()/40)
        noise_img = np.array(255*noise_img, dtype='uint')

        # Preview
        # imshow(noise_img), plt.show()

        os.system(str("mkdir -p ./data/"+"Sample"+str(f_ind).zfill(3)))
        cv2.imwrite("./data/"+"Sample"+str(f_ind).zfill(3) +
                    "/"+char+"_"+str(ind)+"_"+font.getname()[0]+".png", noise_img)
    print(char, ":", freq)
    f_ind += 1
