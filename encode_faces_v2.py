# USAGE
# python encode_faces_v2.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
from imutils import paths
import imutils
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	image_resized = imutils.resize(image, width=750)		# Memory issue
	rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)


	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# Enlarge bounding box
	row_size = image.shape[0]
	column_size = image.shape[1]

	for (top, right, bottom, left) in boxes:
		height = bottom - top
		width = right - left
		top = int(max(top + height * 0.05,0))
		right = int(min(right - width * 0.05,column_size))
		bottom = int(min(bottom - height *0.05,row_size))
		left = int(max(left+width*0.05,0))
		# draw the predicted face name on the image
		cv2.rectangle(image, (left, top), (right, bottom),
			(0, 255, 0), 2)
	boxes = [(top, right, bottom, left)]


	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)



	# print(row_size,column_size,boxes,name)
	# cv2.imshow("CurrentPhoto", image)
	# key = cv2.waitKey(1000)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
