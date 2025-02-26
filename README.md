#tensorflow-Optimizing-Social-Media-Video-Traffic-by-Identifying-Identical-Videos-using-Deep-Learning-Techniques
This repo contains the Python implementation of the TVC'25 paper - [Optimizing Social Media Video Traffic by Identifying Identical Videos using Deep Learning Techniques]. The original TensorFlow/PyTorch implementation can be found [here](https://github.com/chenlepkonyak/optimization-of-osn-video-traffic-by-identifying-identical-videos-using-dl-techniques). 

<div align="center">
  <img src="imgs/pipeline.png" alt="train" width="50%">
</div>

The main requirements are [pytorch]( https://www.tensorflow.org/) (`2.18.0`) and python `3.11`. Some dependencies that may not be installed in your machine are [moviepy](https://pypi.org/project/moviepy/), [kneed](https://pypi.org/project/kneed/), [opencv-python](https://pypi.org/project/opencv-python/), [scikit-learn](https://pypi.org/project/scikit-learn/), [scikit-image](https://pypi.org/project/scikit-image/), [scipy](https://pypi.org/project/scipy/), [pandas](https://pypi.org/project/pandas/) and [numpy](https://pypi.org/project/numpy/). Please install other missing dependencies.

## Get started
1. Download preprocessed datasets

```bash
git clone https://github.com/chenlepkonyak/optimization-of-osn-video-traffic-by-identifying-identical-videos-using-dl-techniques

cd optimization-of-osn-video-traffic-by-identifying-identical-videos-using-dl-techniques

The [HMDB51](https://doi.org/10.1109/ICCV.2011.6126543) dataset is a large collection of realistic videos from various sources, including movies and web videos. The dataset is composed of 6,766 video clips from 51 action categories (such as “jump”, “kiss” and “laugh”), with each category containing at least 101 clips.  
# download datasets_and_experimental_and_analysis_data.tar.gz (173.5MB) contains 1.Setting for HDM51 video Datasets for experiment, 2️.Self-Generated Datasets(server_users_metadata_db) and other 207+ files data accessed through sqlite database is also included which is used for results and analysis data(sever_all_traffic_db).

wget https://zenodo.org/records/14922872/files/datasets_and_experimental_results.tar.gz
tar -xvzf datasets_and_experimental_results.tar.gz
```


2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

## Folder Structure
```bash
📂 optimization-of-osn-video-traffic-by-identifying-identical-videos-using-dl-techniques
│── __init__.py
│── 📂 project_source_code
│   │── 📂 database_modules
│   │   ├── __init__.py
│   │   ├── create_model_database.py
│   │   ├── db_operation.py
│   │── 📂 model_modules
│   │   ├── __init__.py
│   │   ├── generate_dataframe.py
│   │   ├── generate_video_sharing_traffic.py
│   │   ├── generator_UHVID_data.py
│   │── 📂 users_modules
│   │   ├── __init__.py
│   │   ├── generate_server_users_metadata.py
│   │   ├── generator_users_metadata.py
│   │── 📂 utils_modules
│   │   ├── __init__.py
│   │   ├── create_users_view.py
│   │   ├── display_tables_and_views.py
│   │   ├── generate_CSV.py
│   │── 📂 videos_modules
│   │   ├── __init__.py
│   │   ├── generator_server_videos_metadata.py
│   │   ├── generator_users_videos_metadata.py
│   │── sqlite_model.db
|── 📂 csv_data_directory
|── 📂 imgs
│   │   ├── pipeline.png
|── 📂 videos_directory
│── client_model.py
│── server_model.py
│── main.py
│── README.md
│── LICENSE

```

## How to run the code
```bash
#run the main program for initialization and building of extended HMDB51 datasets and all the transactional data, storing it in sqlite database [sqlite_model.db](). 
python main.py 

#Start the Server to run UHVID Validator for optimizing video sharing traffic across social network
python server_model.py

# Run the Client for Video Querying to optimise redundant identical video sharing
python client_model.py 
```

## Visualize main algorithm for generation of UHVID
You can use `generator_UHVID_data.py` to generate the uhvid of the video. You need to have a directory containing video which should be extended HMDB51 datasets with user generated data as [here] (https://zenodo.org/records/14922872/files/datasets_and_experimental_results.tar.gz) . In order to visualize the generation of uhvid, the following command can also be inputted with the video file from the Terminal. The experiments data [csv_data_directory] (https://zenodo.org/records/14922872/files/datasets_and_experimental_results.tar.gz) are analysed for results and discussion on optimizing of OSN traffic using this proposed method. 

```bash
python main.py "CarTwHeeL_PerFecT_cartwheel_f_cm_np1_le_med_0.avi" # replace with any video file
```
Please remember to specify the naming format of your video frames on this [line](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce/blob/master/summary2video.py#L22).

## How to use your own data
We preprocess data by extracting image features for videos and save them to `csv` file. The file format looks like [this](https://zenodo.org/records/14922872/files/datasets_and_experimental_results.tar.gz), as in the setting of HMDB51 for experiments. which need to be updated in the [videos_directory](). [Here](https://github.com/chenlepkonyak/optimization-of-osn-video-traffic-by-identifying-identical-videos-using-dl-techniques/tree/main.py) is the code where we initialize dataset. If you have any problems, feel free to contact me by email or raise an `issue`.

## Citation
```
@article{konyak2025optimzingosntraffic, 
   title={Optimization of Online Social Network video traffic by identifying identical videos using deeplearning techniques},
   author={Konyak, C.Y and Baydeti, N}, 
   journal={The Visual Computer}, 
   year={2025} 
}
```
