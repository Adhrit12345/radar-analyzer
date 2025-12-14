# radar_project

This project classifies the radar data in training mode

To get started



If your virtual environment is not setup, follow the below steps

- Go to "Terminal" menu and open "New Terminal"

- Create virtual environment
```
python -m venv .venv
```

- Activate the virtual environment 

```
.\.venv\Scripts\activate.bat
```

- Install the required pip packages

```
pip install --upgrade pip
pip install -e .
```

- Check if the packages are installed

```
pip list
```

Now, we will start the training. If you dont have your terminal open, 

- Go to "Terminal" menu and open "New Terminal"

- Activate the virtual environment 

```
.\.venv\Scripts\activate.bat
```


Let's start training. Below is the sample structure for training dataset

./training
./training/running/
./training/walking_pet/

To train, run this command (if the training folder path is changed, change it accordingly)

```
python src/radar_project/radar_pipeline.py --mode train --data-root "C:\Users\ADHRIT\rvce\ai-project\training-input" --out-dir "C:\Users\ADHRIT\rvce\ai-project\training-output" --model-dir "C:\Users\ADHRIT\rvce\ai-project\model-dir"
```

To read the training output,run this

```
python src/radar_project/read_training_output.py
```

To predict, run the below command

```
python src/radar_project/radar_pipeline.py --mode predict --predict-folder "C:\Users\ADHRIT\rvce\ai-project\prediction-data" --model-dir "C:\Users\ADHRIT\rvce\ai-project\training-output" --out-dir "C:\Users\ADHRIT\rvce\ai-project\prediction-output"
```

To read the prediction output, run

```
python src/radar_project/read_prediction_output.py
```



To run for real-time data, run the below command

```
python src/radar_project/radar_pipeline.py --mode predict --predict-folder "C:\Users\ADHRIT\rvce\ai-project\real-time" --model-dir "C:\Users\ADHRIT\rvce\ai-project\training-output" --out-dir "C:\Users\ADHRIT\rvce\ai-project\real-time-output"
```

To read the  real-time data prediction output, run

```
python src/radar_project/read_prediction_output.py -p C:\Users\ADHRIT\rvce\ai-project\real-time-output
```

