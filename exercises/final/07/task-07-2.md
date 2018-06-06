# Assignment 07.2 - Expressing the variance

## Assumptions

- n = 1000
- Data:
    1. age
    2. time spent playing with the computer
    3. time spent in facebook
    4. time spend doing sport
- Data has been analysed with PCA

## Question a

What does it mean if a single eigenvector covers 90% of the data variance?

**Answer:** The goal of PCA is to get eigenvectors which explains the most of the variance individually. An eigenvector which explains 90% of the data might be a good choice to drop the other eigenvectors - wether we don't know how many other eigenvectors there are and how many variances they explain. This eigenvector coincide with the direction of the biggest variance.

## Question b

How would you interpret the results if the eigenvector `v1 = [0, 1, 1, 1]^T` covers 85% of the data variance?

**Answer:**  85% of the data is distributed along the 2nd, 3rd and 4th axis in a 1:1:1 ratio. That also means, that 85% of the data has the same value for the first dimension.