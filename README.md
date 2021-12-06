![](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=python&logoColor=white&color=2bbc8a)
![proj_badge](https://img.shields.io/badge/Project-ML-brightgreen)

# BitByteBeat: The Music Genre Classification Bot 🤖 🎶
This repository contains our src code for the COMP432 Machine Learning Project. The purpose of this project was to familiarize the group with working on a machine learning project and to get practical application skills.


## Dataset
Our dataset comes from the [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification/) which is a popular public dataset for music genre recognition (MGR). This is a 1.41 GB folder containing training data that was used to train our models. The particular folder of interest was the Data/genres_original as it contained a collection of 10 genres with 100 audio (.wav) files each. These files were all 30 seconds in length. The 10 genres within this set are:

| Genre       |
| :---------- |
| `blues`     |
| `classical` |
| `country`   |
| `disco`     |
| `hiphop`    |
| `jazz`      |
| `metal`     |
| `pop`       |
| `reggae`    |
| `rock`      |


We would like to note that sample 54 within the jazz set was discarded when training the models as the file was corrupted.

## Project Setup

### Tech Stack

* Recommend python version: 3.6
* Full list of dependencies can be found at [requirements.txt](https://github.com/xkaDachi/COMP432_Music_Genre_Classification/blob/main/requirements.txt)



### Dependency installation
```
pip3 install -r requirements.txt
```


## Jupyter Notebook
- run the `Music Classification.ipynb` file from a Jupyter Notebook

*Note*: The notebook can be previewed directly on GitHub without installing any additional software.



## Running the program from Terminal
- cd into the repo's root directory
- open a terminal emulator of your choice
- run preprocessing
```
python3 preprocessing.py
```
- run both convolution neural network and recurrent neural network models
```
python3 convolutional_neural_network.py & python3 recurrent_neural_network.py
```
- program provides output to the screen as well as data given within the output folder for each task respectively

​

## Authors

- [@Johnny On](https://github.com/xkaDachi)
- [@Chelsie Ng](https://github.com/chelsieng)
- [@Tyler Shanks](https://github.com/HunterShanks)
- [@Siu Ye](https://github.com/SiuYe)

