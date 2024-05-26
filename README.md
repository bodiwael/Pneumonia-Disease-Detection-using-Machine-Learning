# Pneumonia-Disease-Detection-using-Machine-Learning-CNN-

## Child-Pneumonia-Diagnoser

Live Web App: http://child.test.woza.work/

<br>

<img src="http://child.test.woza.work/assets/app_pic.png" width="350"></img>

<br>

This prototype web app uses computer vision to detect viral and bacterial pneumonia on chest x-rays. It’s a triage tool to help doctors identify high risk children fast. The app can accept batches of x-rays in jpg or png format. For each x-ray it outputs three possible conditions:
- No pneumonia
- Viral pneumonia
- Bacterial pneumonia


The model was trained on chest x-rays of children aged 1 to 5. 

Model F1 score: 0.76

The process used to build and train the model is described in this Kaggle kernel:<br>
https://www.kaggle.com/vbookshelf/child-pneumonia-diagnoser-tensorflow-js-web-app/notebook

The dataset used for the training can be found here:<br>
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

All javascript, html and css files used to create the web app are available in this repo.



