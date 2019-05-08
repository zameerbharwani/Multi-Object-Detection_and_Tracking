#/usr/bin/python3
"""
@Author: Zameer Bharwani
"""

import cv2
import argparse
import numpy as np

#better than cv2.putText since it accepts float co-ordinates
from PIL import Image, ImageFont, ImageDraw

def main(min,max,dir):

	"""
	Assumes first frame has no occlusions
	"""
	min = int(min)
	max = int(max)
	complete_annotations = np.loadtxt('./output/mini_test.txt',delimiter =',') # detections for all frames
	font = cv2.FONT_HERSHEY_DUPLEX

	for i in range(int(complete_annotations[:,0].max())): # For every frame

		annotations = np.loadtxt('./output/%s.txt'%(str(i+1)),delimiter=',') # each frame's detections. annotations are stored starting from 1.txt,...,...
		img = cv2.imread("./data/"+dir+"/%s.jpg"%(str(i+min)))

		for j in range(len(annotations)): # for every detection in a given frame
			
			bbox1 = annotations[j,2:] # a given detection's bbox
			cv2.rectangle(img,(int(bbox1[0]),int(bbox1[1])),(int(bbox1[2]+bbox1[0]),int(bbox1[3]+bbox1[1])),(255,0,0),2) #draw the bbox
			ID_1 = annotations[j,1]
			location = (int(bbox1[0]+ (bbox1[2])/2), int(bbox1[1]))
			cv2.putText(img,str(int(ID_1)),location,font, 1,(0,0,0))
			
			for k in range(j+1,len(annotations)): # for every other detection
			
				bbox2 = annotations[k,2:]
				ID_2 = annotations[k,1]
				if i+2 <= (max-min+1):
					output, coord,pred_box = traverse(bbox1,bbox2,i+1,ID_1,ID_2) # traverse the decision tree
				else:
					output,coord,pred_box = None,None,None

				if output is not None and (i+1+min) <= max:

					img2 = Image.open("./data/"+dir+"/%s.jpg"%(str(i+1+min)))
					draw = ImageDraw.Draw(img2)
					font2 = ImageFont.truetype("arial.ttf", 25)
					draw.text(coord,output,(0,0,0),font=font2) 
					img2.save("./data/"+dir+"/%s.jpg"%(str(i+1+min)))

					if pred_box is not None:

						img2= cv2.imread('./data/'+dir+"/%s.jpg"%(str(i+1+min)))
						cv2.rectangle(img2,(int(pred_box[0]),int(pred_box[1])),(int(pred_box[2]),int(pred_box[3])),(0,0,255),2)
						cv2.imwrite('./data/'+dir+'/%s.jpg'%(str(i+1+min)),img2)

		cv2.imwrite("./data/"+dir+"/%s.jpg"%(str(i+min)),img)

def traverse(bbox1,bbox2,frame_numb,ID_1,ID_2):
	"""
	Traverse the decision tree
	"""

	if check_overlap(bbox1,bbox2):

		annotations2 = np.loadtxt('./output/%s.txt'%(str(frame_numb+1)),delimiter=',')
		f2ID = annotations2[:,1] # Frame2 IDs
		f2_ID =[int(iD) for iD in f2ID]

		if int(ID_1) in f2_ID and int(ID_2) not in f2_ID:
			
			x_coord = bbox2[0]+ bbox2[2]/2
			y_coord = bbox2[1]+bbox2[3]
			index = f2_ID.index(int(ID_1))
			bbox = annotations2[index,2:]
			shift_vals = predict(bbox1,bbox)
			bbox2_conv = convert_bbox(bbox2)
			pred_box =[bbox2_conv[i]-shift_vals[i] if shift_vals[i]<=0 else bbox2_conv[i]+shift_vals[i] for i in range(len(shift_vals))]

			return "%d occluded by %d"%(int(ID_2),int(ID_1)), (x_coord,y_coord), pred_box

		elif int(ID_2) in f2_ID and int(ID_1) not in f2_ID:
			x_coord = bbox1[0]+ bbox1[2]/2
			y_coord = bbox1[1]+bbox1[3]
			index = f2_ID.index(int(ID_2))
			bbox = annotations2[index,2:]
			shift_vals = predict(bbox2,bbox)
			bbox1_conv = convert_bbox(bbox1)
			pred_box =[bbox1_conv[i]-shift_vals[i] if shift_vals[i]<=0 else bbox1_conv[i]+shift_vals[i] for i in range(len(shift_vals))]

			return "%d occluded by %d"%(int(ID_1),int(ID_2)), (x_coord,y_coord),pred_box


		elif int(ID_2) not in f2_ID and int(ID_1) not in f2_ID:

			x_coord = bbox1[0]+ bbox1[2]/2
			y_coord = bbox1[1]+bbox1[3]

			return "Re-identification Challenge", (x_coord,y_coord), None

		else:

			return None,None,None

	else:
		return None,None,None


def convert_bbox(bbox):
	"""
	Converts x,y,w,h in to x_min,y_min,x_max,y_max
	"""

	return [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]

def check_overlap(bbox1,bbox2):
	"""
	Checks if 2 boxes overlap. Returns true if overlapping/touching/intersecting
	"""

	[x_min1,y_min1, x_max1,y_max1] = convert_bbox(bbox1)
	[x_min2,y_min2, x_max2,y_max2] = convert_bbox(bbox2)
	
	return ((x_max1 >= x_min2 and x_max2 >= x_min1) and 
	(y_max1 >= y_min2 and y_max2 >= y_min1))

def predict(box1,box2):

	box1 = convert_bbox(box1) # current frame
	box2 = convert_bbox(box2) # future frame
	
	return [box1[0]-box2[0], box1[1]-box2[1], box1[2]-box2[2],box1[3]-box2[3]]

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--min', help ="Lowest frame number in dataset")
	parser.add_argument('--max', help ="Highest frame number in dataset")
	parser.add_argument('--dir', help ="Directory to images inside ./data/")
	parser.parse_args()
	args = parser.parse_args()

	main(args.min,args.max,args.dir)

