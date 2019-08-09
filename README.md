Release Note recognize_faces_video_v2
Date		    : 9-Aug-2019
contributer	: kohtet001@gmail.com

It is based on the code of Andrian Rosebrock from 
https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/

Changes from the original version
- Increase processing speed 5x than the original.
- Improve detection accuracy 2x than the original.
- Require only one dataset picture for each person: Method changed from number of face match counts to minimal distance with similar matrix.
- Implemented face tracking: Correlate bounding boxes of unknown identities with those of known identities from the previous frame.
- Added 3 arguments (search width as '-w', similarity matrix distance threshold as '-s', face movement threshold as '-m').

python encode_faces_v2.py --dataset dataset --encodings encodings.pickle
python recognize_faces_video_v2.py --encodings encodings.pickle --display 1

Step
1) Put one face image of each person to be recognized into each person folder under dataset folder
  example: 1) dataset/Jone/Jone_face.jpg; dataset/Merry/Merry_face.jpg; ... dataset/Steve/Steve_face.jpg;
2) Run the command to extract feature sets from the photos
  python encode_faces_v2.py --dataset dataset --encodings encodings.pickle" 
3) Run the command to recognize faces
  python recognize_faces_video_v2.py --encodings encodings.pickle --display 1
