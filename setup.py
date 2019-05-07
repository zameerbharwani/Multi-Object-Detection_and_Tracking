#!/usr/bin/env python2.7

"""
@Author: Zameer Bharwani
"""

import os
import shutil
import random
import glob
import pandas as pd
from git import Repo
from math import floor


def install_setup():
	"""
	install repo 
	"""
	#git_url = 'https://github.com/kbardool/keras-frcnn.git' 
	#repo_dir = os.path.dirname(os.path.realpath(__file__)) # location of this script
	#Repo.clone_from(git_url, '/home/jzelek/Desktop/Zameer2') # clone git repo into Zameer2
	#os.chdir('/home/jzelek/Desktop/Zameer2/keras-frcnn')
	#os.system("sudo pip install -r requirements.txt")  # download requirements 
	os.mkdir("All_Data")  # create file that will hold all images (located in Zameer2)
	os.chdir('/home/jzelek/Desktop/Zameer2/mcGill/shots')  # change directory to location of images


def transfer_data():
	"""
	moves all images and txt files into a single folder and has the txt files concatenated
	"""

	destination = ('/home/jzelek/Desktop/Zameer2/All_Data')  # where we will place all images
	filenames = []

	for i in range(1,37):
		base_name1 = 'p1-s00'
		base_name2 = 'p1-s0'

		if i < 10:
			source = ('/home/jzelek/Desktop/Zameer2/mcGill/shots/'+ base_name1+str(i)+'/image')
		else:
			source = ('/home/jzelek/Desktop/Zameer2/mcGill/shots/'+ base_name2+str(i)+'/image')

		files = os.listdir(source)

		for f in files:
			shutil.copy2(source+'/'+f, destination)

		os.chdir(source+'/..')
		shutil.copy2(os.getcwd()+'/track-gt.txt',destination)
		os.chdir(destination)
		os.rename('track-gt.txt','annotation_'+str(i)+".txt")
		filenames.append('annotation_'+str(i)+".txt")

	open('complete_annotation.txt', 'a').close()  # this is the file that will hold the concatenated txt

	concatenate(filenames,destination+'/complete_annotation.txt')
	sort(destination+'/complete_annotation.txt')
	separate_annotations(destination+'/complete_annotation.txt')


def concatenate(filenames,destination):
	"""
	Concatenates all txt files, ignores header
	"""
	with open(destination, 'w') as outfile:
		for fname in filenames:
			with open(fname) as infile:
				infile.readline()  # skip header line
				for line in infile:
					outfile.write(line)

def separate_annotations(destination):
	"""
	From the concatenated text file, find all the frame_ids that belong to test/train images and separate them
	"""
	test_image_file_names = [f for f in os.listdir('/home/jzelek/Desktop/Zameer2/All_Data/test_images')]
	train_image_file_names = [f for f in os.listdir('/home/jzelek/Desktop/Zameer2/All_Data/train_images')]
	f=open(destination,"r")
	lines=f.readlines()
	f.close()

	train = open('train_images.txt', 'w')
	

	for i in train_image_file_names:
		for line in lines:
			path_name_train= '/home/jzelek/Desktop/Zameer2/All_Data/train_images/'+i
			if line.split(',')[1] == i:
				train.write(path_name_train +','+ line.split(',')[2] +','+str(float(line.split(',')[3]))+ ',' + str(float(line.split(',')[2]) + float(line.split(',')[4])) +','+str(float(line.split(',')[3]) + float(line.split(',')[5]))+','+'person\n')

	train.close()
				

def sort(name):
	"""
	Creates a dataframe and sorts by frame_ID
	"""
	#  this will order them by frame_id
	df = pd.read_csv(name, sep=",", header=None)
	df.columns = ["", "frame_id", "x", "y", "w", "h", "person_id"]
	df['sort'] = df['frame_id'].str.extract('(\d+)').astype(int)
	df.sort_values('sort',inplace=True, ascending=True)
	df = df.drop('sort', axis=1)
	unique_frames = df.frame_id.unique()  # this is an array
	percent_30 = floor(len(unique_frames)*0.3)
	test_frames=[]

	#  randomly assigns an image to testing 
	while len(test_frames) <= percent_30:
		random_num = random.randint(0,len(unique_frames)-1)
		if unique_frames[random_num] not in test_frames:
			test_frames.append(unique_frames[random_num])

	source = '/home/jzelek/Desktop/Zameer2/All_Data'
	os.chdir(source)
	os.mkdir('test_images')
	os.mkdir('train_images')
	destination2 = ('/home/jzelek/Desktop/Zameer2/All_Data/test_images') 
	destination3 = ('/home/jzelek/Desktop/Zameer2/All_Data/train_images') 

	for i in test_frames:
		shutil.move(str(i),destination2)


	train_frames =[i for i in os.listdir(destination3+'/..')]
	for j in train_frames:
		if j.endswith('.jpg'):
			shutil.move(j,destination3)


if __name__ == "__main__":

	install_setup()
	print ("\nInitial Set up is complete. Now organizing data\n")
	transfer_data()
	print ("All done. Data ready for neural network training!")







