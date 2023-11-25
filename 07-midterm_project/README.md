# **Bank Marketing Outcome Prediction**

## Project Description

This project endeavors to construct a robust classification machine learning model designed to forecast the likelihood of a bank client subscribing to a term deposit. The predictive analysis is rooted in the examination of direct marketing campaigns conducted by a Portuguese banking institution, where phone calls served as the primary communication medium. The complexity of the task is underscored by the necessity for multiple interactions with the same client to discern whether they would ultimately subscribe ('yes') or decline ('no') the bank term deposit.

The overarching objective is to leverage machine learning algorithms to decipher the key factors influencing a customer's decision to subscribe to the product. By harnessing the power of predictive modeling, the aim of this project is to unearth the nuanced patterns and insights within the dataset, offering valuable guidance to optimize future marketing strategies.

## Data

The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

Data source: https://archive.ics.uci.edu/dataset/222/bank+marketing

## Exploring the model in a notebook

The notebook is available in this repository: **notebook.ipynb**

## Dependencies

1. Install the requirements with pipenv `pipenv install`

## Web Service Deployment

1. Run the prediction python script `python predict.py`

2. With the Flask application running, we can make HTTP requests to port 1313 by running the predict-test python script `predict-test.py`. Adjust the parameters as you wish.
   
## Docker

1. Build the docker image `docker build -t marketing-prediction .`
2. Run the docker image `docker run -it -p 1313:1313 marketing-prediction:latest`
3. With the docker container running, we can make HTTP requests to port 1313 by running the predict-test python script `predict-test.py`. Adjust the parameters as you wish.
