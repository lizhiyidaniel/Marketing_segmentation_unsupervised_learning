# Marketing_segmentation_unsupervised_learning
Perform marketing segmentation for advertising by unsupervised learning techniques: clustering (k-means), principal component analysis and auto encoders

[View on Notebook](https://github.com/lizhiyidaniel/Marketing_segmentation_unsupervised_learning/blob/main/Marketing_Segmentation_Unsupervised_Learning.ipynb)

# Understand Marketing Segmentation

Why Marketing is essential? - Marketing is crucial for the growth and sustainability of businesses - Marketing could help to build the company's brand, engage customers, grow revenue and increase sales

1.  Growth: empowering business growth by reaching new customers
2.  Education: educating and communicating value porposition to customers
3.  Drive sales: driving sales and traffic to products/services
4.  Engagement: engaging customers and understand their needs

Why Market Segmentation is important?

-   One of the key pain points for marketers is to know customers and identify their needs
-   By understanding customers, marketers could launch a targeted marketingg campaign to tailor for specific needs
-   Data Sciene could be used to perform market segmentation

Task: Launch targeted ad marketing compaign by dividing customers into distinctive groups data: banking data about customers for past 6 months

Data Source: <https://www.kaggle.com/arjunbhasin2013/ccdata>

Data Information: This case requires to develop a customer segmentation to define marketing strategy. The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.

# What is K-MEANS?

K-means intuition: 
- K-means is an unsupervised learning algorithm (clustering) 
- K-means works by grouping some data points togetheer (clustering) in an unsupervisedd fashion 
- The algorithm works by grouping observations with similar attributes values together by measuring the Euclidian distance between points

K-means algorithm steps 
1. Choose number of clusters "K" 
2. Select k random points from the data as centroids 
3. Assign all the points to the closest cluster centroid 
4. Recompute the centroids of newly formed clusters 5. Repeat steps 3 and 4

Stopping Criteria for K-Means Clustering 
1. Centroids of newly formed clusters do not change 
2. Points remain in the same cluster 
3. Maximum number of iterations are reached

How to select the optimal number of clusters? Elbow method 
- The elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help find the appropriate number of clusters in a dataset. 
- If the line chart looks like an arm, then the "elbow" on the arm is the value of k that is the best. 
- the cluster value where this decrease in inertia value becomes constant can be chosen as the right cluster value for our data.


# APPLY PRINCIPAL COMPONENT ANALYSIS AND VISUALIZE THE RESULTS

What is Principal Component Analysis: 
- PCA can be an excellent way to understand which features are more relevant for a particular machine learning problem. 
- PCA is a practical mathematical approach for reducing data complexity. It finds linear combinations of the input fields that best represent the statistical variation in the complete set of areas. The data components are orthogonal to and unrelated to one another. 
- We can use PCA to select the most relevant data features.

# Use autoencoders to perform data encoding

Tuition behind autoencoders: 
- Auto encoders are one of artificial neural networks that are used to perform data encoding (representation learning) 
- Auto encoders use the same input data for the input and output
