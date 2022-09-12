#!/bin/python3
#This pulls a video from a .rar, converts it to a multidimensional numpy array, then sends it to the autoencoder for processing.

from os import walk as walk_dir, remove as rmv
from unrar import rarfile

import numpy as np
import cv2

import autocompress as ac

# Globals
tmp = "/mnt/video_rfs"

def read_dir(path, suffix):
	for (dirpath, _, files) in walk_dir(path):
		files = [dirpath + "/" + x for x in files if x.endswith(suffix)]
		return files

def main():
	global tmp
	files = read_dir("testing_data", ".rar")
	
	# for file in files:
	while (True):
		file = files[0]
		# Skip bad files
		if (not rarfile.is_rarfile(file)):
			print("Skipping bad file: %s" % file)
			continue
		
		rar = rarfile.RarFile(file)
		print("=== RAR FILE: %s ===" % file)
		videos = [x for x in rar.namelist() if x.endswith(".avi")]
		
		for vid in videos:
			print("Loading %s..." % vid)
			# Extract to tmp and read
			rar.extract(vid, tmp)
			vid_cv = cv2.VideoCapture(tmp + "/" + vid)
			
			# Delete tmp file
			rmv(tmp + "/" + vid)

			# Work on set of video frames
			i = 0
			voxel = np.ndarray([0, *ac.VOXEL_SHAPE[1:]])
			while (vid_cv.isOpened()):
				# On every voxel
				if (i != 0 and (i % ac.VOXEL_SHAPE[0]) == 0):
				
					queue_voxel(voxel)
					voxel = np.ndarray([0, *ac.VOXEL_SHAPE[1:]])
				
				# Read frames otherwise
				ret, frame = vid_cv.read()
				if ret:
					frame.resize(1, *ac.VOXEL_SHAPE[1:])
					voxel = np.concatenate((voxel, frame), axis = 0)
				else:
					print("Could not read frame %i" % i)
					break
				
				# Increment frame counter
				i += 1

			# Release after
			vid_cv.release()

voxels = np.ndarray([0, *ac.VOXEL_SHAPE])
def queue_voxel(voxel):
	global voxels

	reshaped = np.reshape(voxel, [1, *ac.VOXEL_SHAPE])
	reshaped = np.divide(reshaped, 255)
	voxels = np.concatenate((voxels, reshaped), axis = 0)

	#print(voxels.shape)

	# Process batch of 1 video voxels
	if (voxels.shape[0] == 1):
		
		ac.process(voxels)
		voxels = np.ndarray([0, *ac.VOXEL_SHAPE])


if (__name__ == "__main__"):
	main()
