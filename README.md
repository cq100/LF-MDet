# LF-MDet
Codes for ***Low-rank Multimodal Remote Sensing Object Detection with Frequency Filtering Experts***

Xu Sun, Yinhui Yu*, and Qing Cheng


## Update
- [2024/7] This code will be released soon.



### ⚙ Network Architecture

<img src="image//architecture.png" width="90%" align=center />


## 🌐 Usage

**1. Virtual Environment**
```
conda env create -f environment.yml
```

**2. Data Preparation**

The convert_yolo.py script is designed to facilitate the conversion of object detection annotations from the OpenMMLab format to the Ultralytics YOLO format.


**3. LF-MDet Training**

Run 
```
source ~/.bashrc
conda activate openmmlab
which python
nohup python3  {config_path}\
    --work-dir  {checkpoint_path}\
    --gpu-ids 0 > {log_path}.log 2>&1 &
``` 
and the trained model is available in ``'./checkpoints/'``.


**4 LF-MDet Testing**

Run 
```
source ~/.bashrc
conda activate openmmlab
which python
python3  {config_path}\
    --work-dir  {checkpoint_path}\
    --eval bbox \
    --gpu-ids 0
``` 


**5 The visualizing detection results of our approach on the VEDAI and DroneVehicle datasets.**

<img src="image//results1.png" width="90%" align=center />


<img src="image//results2.png" width="90%" align=center />
