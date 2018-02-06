# ML_DSIP_Proj
This project was done using Anaconda for windows , with python 3.5.
it includes python code files and jupyter notebooks (for calling the code)
the following additional pre-requisits are required:
- tensorflow framework (for facenet cnn )
- Dlib (For face detection and cropping functionality but it's not mandatory if the dataset is already cropped)
- FaceNet pre-trained model (available here: https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit)
- Dlib pre-trained model for face detection and landmark detection (available here: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 ) ;not mandatory if the dataset faces are already cropped
the notebook LatestVersion contains code snippets to call the main functions of the project, which also have queriable help.

It's possible to use any cropped dataset, the datasets I have used are:
- AT&T(ORL Dataset) available here: http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip
- Yale Dataset
LFW available here:http://vis-www.cs.umass.edu/lfw/
