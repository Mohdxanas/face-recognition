# face-recognition
something new
#how to run
1. Install dependencies: `pip install opencv-python opencv-contrib-python numpy Pillow`
2. Place training images in `images/`, with a subfolder for each person's name.
3. Update file paths in `faces-train.py` and `Face Recognition.py` to match your local setup.
4. Run `python faces-train.py` to train the LBPH recognizer and generate `trainner.yml` and `labels.pickle`.
5. Run `python "Face Recognition.py"` to start real-time face recognition via webcam. Press **q** to quit.
