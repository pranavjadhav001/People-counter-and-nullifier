# People-Counter-and-Nullifier
Counts people in a real time video and shows bbox around them , Also you can nullify or ignore some people from count like employees or guards.<br />
It can be used in situations where you want to count only customers in a shop and ignore the employees or guards present.<br />
Now it is trained , to detect people other than police or uniformed men/women.
## How it works
- First, we take each frame of video and using any object detection model. Here, I used SSD mobilenet , detect all the objects present.
- Since, we all only concerned with people only, we narrow it down and get the coordinates of all people.
- Create bounding box around all people and change the count status showing on screen accordingly.
- Now crop these individual bbox and sent as input through to the next model which is classification.
- To check whether the person is employee or customer based on their appearance.
- Ignore all employees and don't count and bbox them.

#### How it looks 
![alt text](https://github.com/pranavjadhav001/People-counter-and-nullifier/blob/master/pictures/Capture.PNG)

## How to use
python final.py -m trained_classification_models -v path to video_file 

## Improvements

1.Frame rate can be improved.<br />
2.yolo for object detection.<br />
3.classification model can be optimised for high performance.<br />
4.Instead of predicting from entire new model , use features from object detection and predict based on them.<br />


