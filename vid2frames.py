import cv2
import os


def vid2frames():
	os.mkdir("Frames")
	vidcap = cv2.VideoCapture('sort_video.MP4')
	success, image = vidcap.read()
	count = 1
	success = 1
	while success:
		cv2.imwrite(os.getcwd()+'/Frames'+'/%#01d.jpg' % (count), image)
		success, image = vidcap.read()
		count += 1


if __name__ == '__main__':
	vid2frames()

	
# script from stackoverflow
