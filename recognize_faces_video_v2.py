# USAGE
# python recognize_faces_video_v2.py --encodings encodings.pickle
# python recognize_faces_video_v2.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-w", "--search-width", type=int, default=750,
	help="face search width, smaller faster but less accurate")
ap.add_argument("-s", "--similarity-matrix-distance", type=int, default=0.48,
	help="Similarity Matrix Distance Threshold")
ap.add_argument("-m", "--face-movement-threshold", type=int, default=2000,
	help="face movement threshold")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# Creating cummulative distance
# closeness_record = np.zeros(len(data["names"]))
# print(closeness_record.shape)
max_value = 0;


# loop over frames from the video file stream
prev_names_boxes = []
while True:
	
	# grab the frame from the threaded video stream
	frame = vs.read()
	start = time.time()
	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(rgb1, width=args["search_width"])		# Processing speed
	r = frame.shape[1] / float(rgb.shape[1])
	

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])


	# Contributed by kohtet001@gmail.com
	# face_recognition with orignial size face
	boxes_enlarged = []
	for (top, right, bottom, left) in boxes:
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		boxes_enlarged.append((top, right, bottom, left))

	encodings = face_recognition.face_encodings(rgb1, boxes_enlarged)
	# Initializing variables
	names = []
	names_boxes = []
	idx=0;



	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings

		name = "Unknown"
		distances = face_recognition.face_distance(data["encodings"], encoding)

		# Feature distance calculation
		min_distance = min(distances)



		# Contributed by kohtet001@gmail.com
		# Find minimal feature
		min_distance_idx = np.where(distances == min_distance.min())
		if min_distance < args["similarity_matrix_distance"]:
			name = data["names"][min_distance_idx[0].item(0)]
		else:
			for i in range(0, len(prev_names_boxes)):
				# Get previous box
				tmp_list = prev_names_boxes[i][0:4]
				face_movement = (sum(np.array(list(boxes[idx])) - np.array(tmp_list)) ** 2) * (r ** 2)
				# print(face_movement)
				if(face_movement < args["face_movement_threshold"]):	#compare with current box
					name = prev_names_boxes[i][4]
		idx = idx+1


		# update the list of names
		names.append(name)
	idx = 0
	end = time.time()
	# print(end-start)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):

		# Contributed by kohtet001@gmail.com
		names_boxes.append([top, right, bottom, left, name])	#Save previous names and boxes


		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)





		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces t odisk
	if writer is not None:
		writer.write(frame)

	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(10) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	if (len(boxes) > 0):
		prev_names_boxes = names_boxes[:]
	else:
		prev_names_boxes = []

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()

