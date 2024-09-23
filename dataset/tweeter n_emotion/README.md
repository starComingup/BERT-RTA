Emotion detection from text is one of the challenging problems in Natural Language Processing. 
The reason is the unavailability of labeled dataset and the multi-class nature of the problem. 
Humans have a variety of emotions, but it is difficult to collect enough records for each emotion and hence the problem of class imbalance arises. 
Here we have a labeled data for emotion detection and the objective is to build an efficient model to detect emotion.


Statistic:
- total lines: 40000
- sentiment distribution

| neutral(?) | anger(0) | boredom(1) | empty(?) |
|:----------:|:--------:|:----------:|:--------:|
|    8638    |   110    |    179     |   827    |

| enthusiasm(2) | fun(3) | happiness(4) | hate(5) |
|:-------------:|:------:|:------------:|:-------:|
|      759      |  1776  |     5209     |  1323   |

| love(6) | relief(7) | sadness(8) | surprise(9) | worry(?) |
|:-------:|:---------:|:----------:|:-----------:|:--------:|
|  3842   |   1526    |    5165    |    2187     |   8459   |