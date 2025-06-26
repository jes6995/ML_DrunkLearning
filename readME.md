# Computer Vision Impairment Detection Model

## How to Run:

In order to run the RNN model (our best model), you can find the file ‘run_model.py’ in the ‘best_rnn_model’ folder.
The model has already been trained so it does not need to be trained again. The trained model has been saved inside
the mentioned folder. When running the file ‘run_model.py’, it will prompt you to enter the path to the folder
containing the test videos. Enter the path, but you must make sure that there are no quotations around the path.
The file will now continue running. It will take some time to process each video in the folder (around 20 seconds
for each video), so please be patient. Once it is finished, the file will output an array of predictions for videos
as either ‘drunk’ or ‘sober’.


## How to train the RNN model:

To train the RNN model, you must run the ‘train_model.py’ file in the ‘best_rnn_model’ folder. It will prompt you to
enter the path of the folder containing the CSV files (these files contain data collected from the videos). You must
enter the path of the ‘quad_data’ folder which is in the ‘best_rnn_model\data’ directory. This folder contains the 
data from the 800 data points. Make sure that there are no quotations around the path. The file will begin training 
the model now, and when finished it will save the model files.


## How to process videos for data:

To obtain data from the videos, you must run the ‘dui.py’ file in the ‘best_rnn_model’ folder. It is not recommended
to run this file as you will have to run all of the sober and drunk videos, then properly label them and place them
in the correct folder. You will be prompted to enter the path to the folder containing the test videos. Remove any 
quotations around the path before entering it. After this, the file will process each video. This will take a long
time, so do not do this unless you are willing to wait. Once done, go to the data folder and then place the CSV 
files in a folder of your choice. Rename the CSV files to ‘xxx_data’ if it is sober, and ‘xxx_data_drunk’ if it is 
drunk data. The xxx would be the name of the file type (deviation, rate of change, normalized).

## Additional

If you are running into any errors while attempting to run the model, it is recommended to retrain the model, then 
attempt to run the model again.
