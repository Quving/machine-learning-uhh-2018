# Assignment 10
---
# a) Introduction

> Give a short overview over the data-set. Explain the features and their types.


We have chosen the Global Terrorism Database, Version 2 [1] by the START consortium. It contains information on more than 170,000 Terrorist Attacks. 

> "The Global Terrorism Database (GTD) is an open-source database including information on terrorist attacks around the world from 1970 through 2016 (with annual updates planned for the future). The GTD includes systematic data on domestic as well as international terrorist incidents that have occurred during this time period and now includes more than 170,000 cases. The database is maintained by researchers at the National Consortium for the Study of Terrorism and Responses to Terrorism (START), headquartered at the University of Maryland." 

More information can be found [here on Kaggle](https://www.kaggle.com/fabsta/terrorism-in-europe-and-germany/data).


# b) Data analysis

> Describe your findings of the data analysis along with the basic statistics of the data-set: means, medians, variances, empirical probabilities etc. Also identify class imbalances, non-normalized features, and include statistics for appropriate subgroups.
> 

The dataset contains 135 columns in total from terrism attacks between 1970 and 2017, excluding 1993 because of a data lost. Some of the columns contain string data or categorial numbers. 

For additional data, we have added an additional column which contains an assigned ID to every `gname` – the terrorist's group name to every attack. Also, we have added a column which contains a binary value if the nationality of the fatalities is the same as the country's ID to compare it later with Pearson correlation coefficient.

We have calculated the mean, median and variances of all numeric columns, as well as the probability for standard deviation with Shapiro-Wilk test. As expected, none of the columns is deviated like a standard deviation. The exact values are listed in appendix. We than have plotted the values as histograms and scatter plots.

As third step, we have then plotted the variables in pairs as heatmaps to get an idea for potential correlations. It is apparent that the most attacks happened in the last 10 years, according to figure X below. Also, the nationality of fatalities is most of the time the country of the attack, which is visible as a clear coloured line in figure Y.

![figure-corr: eventId and iYear](https://i.imgur.com/yIn2Tqc.png)


![](https://i.imgur.com/WUixBuY.png)


## Data columns

, which we have extended by two additional columns for.

## Interesting features and subgroups

> Identify potentially interesting features and subgroups of features. Are the features which are correlated with each other? Can you identify groups of uncorrelated features, for example by clustering features with respect to the correlation coefficient as a similarity function? Which features seem to have a lot of explanatory or predictive potential?
> 
---
### Subgroup 1

The following plot shows the distribution of terrorism attacks over countries. The countries are decoded as id. As shown in the plot below, there are three extremes marked. Interval of country ids 90-100 has 27.4k attacks, 140-150 19.2k attacks and 0-10 15.4k attacks.


![](https://i.imgur.com/GYL0KDS.png)

The plot below visualizes the location of the terrorism attacks globally given by its longitude and latitude. A remarkable finding is the high density of attacks in the area of middle-west-asian, middle asian and at the west-coast of middle america. 
![](https://i.imgur.com/4HRcmyh.png)

### Relation of the two plots: 
At the histogram it can be seen that the most attacks occurs globally in the country id from 90 to 100. (Please take note, the interval correlates with the chosen bin size.) Countries that can be looked up with the mentioned id's are *Hungary, Iceland, India,  Indonesia, Iran, Iraq, Ireland, Israel, Italy, Ivory Coast*. Since the id's are given to countries according to the alphabetical order of the country name, countries can also considered into the local density while there are no attacks. 

### Correlations

We have calculated the Pearson correlation coefficient for each numeric column. For this we have replaced every incidents of `NaN` with the column's average mean. With this, we have calculated then a `n x n` matrix, where `n`is the number of columns. Columns which contain strings were left out (and have a coefficiency of 0 in our matrix). We then have plotted the matrix with Matplotlib and explored the peaks of the coefficent matrix and have chosen up to five correlations which have a semantical meaning (see figure X). For example we had a **strong correlation of 0.982** for the columns *targtype* and *targsubtype*, but this contains no semantic value, since every target type is defined also by it's target subtypes.

![Figure: interactive selection of coefficients](https://i.imgur.com/iP83XdE.png)

Therefore, we have chosen the following positive and negative coefficients as interesting and worth to explore in more detail, whose *coef = < -0.3* or *coef = > 0.3*:


| Column 1 | Column 2 | Coefficient | Explanation
| - | - | - | - |
|iyear|longitude|0.54| The year intends slightly, where the terrorist attack might occur |
|extended|ishostkid|0.36| |
|attacktype1|weaptype1|0.65| There is a tendency, where the type of attack leads to the chosen weapon type. |
|nkill|nkillus|0.44||
|nkill|nkillter|0.35||
|nkill|nwound|0.46||
|nkillus|nwound|0.75| The chance of having a US fatality in the number of injuries. |
|nkillter|nwoundte|0.37|
|natlty1|country|0.59| The nationality of the target/victim might be also the country of attack.|
|natlty_differ|country|0.38||
|ndays|hostkidoutcome|-0.42||
|ndays|nreleased|0.33||

Additionally, we have tested with the same method the following hyptheses, found in the data:

## Clustering

> Run at leat one clustering method on the data; or explain why clustering won’t help on your data-set. If necessary, preprocess the data. You may focus on specific feature subgroups for clustering (this means that you only use some of the features to cluster your data). Interpet the found clusters and identify meaningful clusters.
> 

## Reduction of dimensions

> Run at least one dimensionality reduction method; or explain why this won’t work on your data-set. Can you simplify the problem with dimesnionality reduction methods? What might be a good choice for the diemnsion parameter?

## Misc

> Are there any other aspects of the data that you find interesting?

# Conclusion

> Provide concluding remarks about your findings. Describe what your findings suggest for the predictive modeling task in Assignment 11.

# Manipulation on Dataset
- Delete **AttackType2\*, AttackType3\*** (broken data).
- Fill NaN's with mean of the list to prevent data loss.


# References

> When you use special algorithms, techniques, or software libraries, provide references to the papers, yournals, and websites.

[1] START CONSORTIUM: Global Terrorism Database, Version 2,  https://www.kaggle.com/START-UMD/gtd (last visit on 29th of June, 2018, 20:02)
[2] DOCUMENTATION FOR GTD DATABASE https://www.start.umd.edu/gtd/downloads/Codebook.pdf

# Appendix A
| id | columname | mean | median | variance | W | normality |
|-|-|-|-|-|-|-|
| 0 | eventid | 200177632735.86 | 200712130009.50 | 1727753934559308032.00 | 0.82 | 0.00 |
| 1 | iyear | 2001.71 | 2007.00 | 172.77 | 0.85 | 0.00 |
| 5 | extended | 0.04 | 0.00 | 0.04 | 0.20 | 0.00 |
| 7 | country | 132.53 | 98.00 | 12734.63 | 0.71 | 0.00 |
| 9 | region | 7.09 | 6.00 | 8.70 | 0.90 | 0.00 |
| 13 | latitude | 23.40 | 31.13 | 345.53 | 0.92 | 0.00 |
| 14 | longitude | 26.35 | 42.24 | 3337.68 | 0.91 | 0.00 |
| 20 | crit2 | 0.99 | 1.00 | 0.01 | 0.06 | 0.00 |
| 21 | crit3 | 0.88 | 1.00 | 0.11 | 0.38 | 0.00 |
| 23 | alternative | 1.29 | 1.29 | 0.07 | 0.31 | 0.00 |
| 28 | attacktype1 | 3.22 | 3.00 | 3.58 | 0.76 | 0.00 |
| 34 | targtype1 | 8.40 | 4.00 | 44.20 | 0.85 | 0.00 |
| 36 | targsubtype1 | 46.87 | 39.00 | 904.80 | 0.93 | 0.00 |
| 40 | natlty1 | 127.73 | 107.00 | 7736.05 | 0.79 | 0.00 |
| 42 | targtype2 | 10.20 | 10.20 | 1.94 | 0.29 | 0.00 |
| 44 | targsubtype2 | 28.37 | 28.37 | 41.16 | 0.24 | 0.00 |
| 50 | targtype3 | 9.88 | 9.88 | 0.20 | 0.05 | 0.00 |
| 52 | targsubtype3 | 55.07 | 55.07 | 4.01 | 0.04 | 0.00 |
| 71 | claimed | 0.03 | 0.00 | 0.80 | 0.19 | 0.00 |
| 75 | claimmode2 | 7.22 | 7.22 | 0.02 | 0.02 | 0.00 |
| 78 | claimmode3 | 7.09 | 7.09 | 0.01 | 0.01 | 0.00 |
| 80 | compclaim | -6.42 | -6.42 | 0.48 | 0.15 | 0.00 |
| 81 | weaptype1 | 6.43 | 6.00 | 4.63 | 0.60 | 0.00 |
| 85 | weaptype2 | 6.74 | 6.74 | 0.34 | 0.26 | 0.00 |
| 87 | weapsubtype2 | 10.67 | 10.67 | 3.49 | 0.29 | 0.00 |
| 89 | weaptype3 | 6.87 | 6.87 | 0.05 | 0.07 | 0.00 |
| 91 | weapsubtype3 | 11.51 | 11.51 | 0.62 | 0.07 | 0.00 |
| 93 | weaptype4 | 6.24 | 6.24 | 0.00 | 0.00 | 0.00 |
| 95 | weapsubtype4 | 10.79 | 10.79 | 0.03 | 0.00 | 0.00 |
| 98 | nkill | 2.39 | 1.00 | 121.02 | 0.15 | 0.00 |
| 99 | nkillus | 0.05 | 0.00 | 22.15 | 0.00 | 0.00 |
| 100 | nkillter | 0.48 | 0.00 | 10.69 | 0.08 | 0.00 |
| 101 | nwound | 3.20 | 0.00 | 1092.44 | 0.04 | 0.00 |
| 103 | nwoundte | 0.10 | 0.00 | 1.25 | 0.04 | 0.00 |
| 109 | ishostkid | 0.06 | 0.00 | 0.20 | 0.17 | 0.00 |
| 113 | ndays | -31.89 | -31.89 | 613.75 | 0.10 | 0.00 |
| 122 | hostkidoutcome | 4.62 | 4.62 | 0.24 | 0.27 | 0.00 |
| 124 | nreleased | -28.72 | -28.72 | 188.78 | 0.21 | 0.00 |
| 130 | INT_LOG | -4.58 | -9.00 | 20.64 | 0.65 | 0.00 |
| 131 | INT_IDEO | -4.51 | -9.00 | 21.44 | 0.67 | 0.00 |
| 132 | INT_MISC | 0.09 | 0.00 | 0.34 | 0.22 | 0.00 |
| 133 | INT_ANY | -3.98 | 0.00 | 22.01 | 0.68 | 0.00 |
| 135 | natlty_differ | 0.13 | 0.00 | 0.11 | 0.39 | 0.00 |
| 136 | gname_ids | 2477.41 | 3050.00 | 1070155.58 | 0.79 | 0.00 |
