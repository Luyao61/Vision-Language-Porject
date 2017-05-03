# Vision-Language-Project

This is an course project for [CS 6501-004: Vision & Language][1]

This project reimplements some models using pytorch from the follwoing papers:

-[Exploring Models and Data for Image Question Answering][2]

-[Stacked Attention Networks for Image Question Answering][3]

-[Ask Your Neurons: A Neural-based Approach to Answering Questions about Images][4]

## model VIS+LSTM
![Model1 architecture](https://github.com/Luyao61/Vision-Language-Project/blob/master/graphes/Model_1_hd.png)
## model VIS+LSTM-2
![Model2 architecture](https://github.com/Luyao61/Vision-Language-Project/blob/master/graphes/Model_2_hd.png)
## model SANS
![Model3 architecture](https://github.com/Luyao61/Vision-Language-Project/blob/master/graphes/Model_3_hd.png)

## Setup

Requirements:
- [python]
- [Pytorch][6]
- [Keras]

## Usage

### Download the [MSCOCO][7] train+val images & Download the [VQA][5]
`sh data/download_data.sh`

### Extract image features 
place then in `features\`
`python get_features.py`
`python get_features_SAN.py`

### Training
run train file to train each model

### Testing
run jupyter notebook to see the results

## Expriment Results
![correct answer 1](https://github.com/Luyao61/Vision-Language-Project/blob/master/graphes/correct_model2.png)
![wrong answer 1](https://github.com/Luyao61/Vision-Language-Project/blob/master/graphes/wong_model2_1.png)
![correct answer 2](https://github.com/Luyao61/Vision-Language-Project/blob/master/graphes/correct_model_san.png)
![wrong answer 2](https://github.com/Luyao61/Vision-Language-Project/blob/master/graphes/wrong_model_san.png)


  


[1]: http://www.cs.virginia.edu/~vicente/vislang/
[2]: https://arxiv.org/abs/1505.02074
[3]: https://arxiv.org/abs/1511.02274
[4]: https://arxiv.org/abs/1505.01121
[5]: http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/
[6]: http://pytorch.org/
[7]: http://mscoco.org/
