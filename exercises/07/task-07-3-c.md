## Question
After splitting the dataset into training and test sets, the demo runs a PCA on the training
set. Is the initial choice of nc = 150 principal components a good choice?

## Answer
The choice of 150 is good, because the data-variance of the most frequent is already quite high. By increasing
n_components the data-variance isn't growing that much. (See task-07-3-c_plot.png)
