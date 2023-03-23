# Final Project of the course "Machine Learning"
## Team: The Ancient Octopi
### Team members:
- Ivan Gurev
- Nikolay Kotoyants
- Gleb Mazanov
- Anna Iliushina
- Viacheslav Naumov

## Topic of the project
Contrastive Learning for Event Sequences with Self-Supervision on multiple domains

This project is devoted to classifying users for higher education, using transaction and clickstream data.

Data is taken form [Data Fusion Contest 2022](https://ods.ai/competitions/data-fusion2022-education)

# Overview
## Data preprocessing
From the dataset we've got following files:
1. `transactions.csv` - transactions data
2. `clickstream.csv` - clickstreams data
3. `train_matching.csv` - exact correspondence between client id in transaction and clickstream data.
4. `train.csv` - target data

After preprocessing resulting features for clickstreams and transactions were saved [here](https://drive.google.com/drive/folders/1zXjzLf0uGyT2mM8po8LgCz2z6Vvy2Gzi?usp=share_link)

## Model
This is neural network based solution. We use [pytorch-lifestream](https://github.com/dllllb/pytorch-lifestream) library to simplify work with sequences data.

We create neural network to encode transaction sequences and clickstream to vector representation. Matching and ranking based on L2 distance between transaction sequence embedding and clickstream embedding.

Network include two stages:

* TrxEncoder, which transform each individual transaction or click into vector representation
* SequenceEncoder, which reduce sequence of vectors from TrxEncoder into one final vector

We have used 3 different SequenceEncoders:
* CoLES
* Agg Baseline
* RandomEncoder

We have used `RandomForest` as a final classifier. 

Final models are saved in **models** folder

## Results

We have obtained following results for the corresponding models:

|Metric| Transactions_agg | transactions_random | transactions_coles|clickstreams_agg|clickstreams_random|clickstreams_coles|
|-------:| ------------- |:-------------:| -----:|-----:|-----:|-----:|
|Acc_score|0.780|0.694|0.764|0.644|0.694|0.729|
|Prec_score|0.794|0.728|0.790|0.734|0.728|0.729|
|Recall|0.950|0.926|0.928|0.801|0.926|0.999|
|F1|0.865|0.815|0.854|0.766|0.815|0.843|
|ROC\AUC|0.621|0.499|0.609|0.513|0.499|0.504|

