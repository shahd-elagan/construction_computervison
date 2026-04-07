# construction_computervison
AI-Powered Industrial Machine Utilization &amp; Activity Monitoring Dashboard A real-time computer vision system using YOLOv5 and LSTM to track machinery, analyze operational status (Active/Idle), and predict activities (Digging, Loading, etc.). Features a Streamlit dashboard, PostgreSQL integration for logging, and automated JSON session exports.

# explnantion of  files in the project :

1-roboflow file >> this code was used to train the yolov5 model on the construction data set found in roboflow , the data has 3 classes ( 0 > closed white cars , 1 > deep truch , 2 > the extractavor ) 



2- testing1 > this code was used to test the accuracy of the pretrained model on the items in videos not images 


3- testing2 > this code was used to detect whether the objects are static or moving so it was detection for idle and active mode 


4-savingposes> the poses ( diging , loading , dumping , swinging ) need to be collected manually  so i used a youtube videos to cut every 30 second frames into a clip to classify each the training was saved in a .pth file 


5-database > the code for creation a table in postgress 


6-app.py > this was the final code containg the yolo pretrained file , the pth file for poses and the streamlit as well as part responisble for generating a json file 




# interface options 

the interface containt a button for uploading video , stopping the run , saving to json and it shows a total of idle and active seconds 


