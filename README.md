# ML4_Face-Mask-Detection

This is a trained Machine Learning Model that can detect whether a person or multiple people is/are wearing or not not wearing a mask.

A] Requirements:
1) Python (recommended version 3.7.6 and above)
2) pip install -r requirements.txt   (Please run your command prompt/terminal as administrator) 

B] After downloading the project, the first thing you will need to do is to download the .weights file:
1) Run the python script 'download_weights.py' (Preferably use a command line so that you can see the download progress)
2) Ensure that a file named 'yolov3_training_last.weights' is now present in your project directory and it has a size of around 240mb

C] For Detection of a Single Image
1) Run the python script 'Detection_Image.py' and the image will load on your screen 
2) To try a different image, change the image file name in the python script  (Defualt image name is 'Test_Image (1).jpg') 

D] For Detection using a Web Cam
1) Run the python script 'Detection_Webcam.py' and your webcam will open
2) Please allow Python to access WebCam if asked so by your anti-virus software

E] For a more interactive experience
1) Click on 'Face Mask Detection.bat' and wait for Chrome to open
2) The website will now load...In case the website fails to load, wait for a few seconds and hit the refresh button

For both points D and E, press the 'Escape' button to close the WebCam

