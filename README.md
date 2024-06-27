# Musical Instruments Reviews Sentiment Analysis

## Overview

This project performs sentiment analysis on musical instrument reviews using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. The dataset used consists of reviews for musical instruments, and the analysis includes calculating sentiment scores and visualizing the results.

## Prerequisites

Before running the project, ensure you have the following libraries installed:

- pandas
- matplotlib
- seaborn
- nltk

You can install these packages using pip:

```bash
pip install pandas matplotlib seaborn nltk
```

## Dataset

The dataset used in this project is `Musical_instruments_reviews.csv`. The first 1000 entries of this dataset are used for the analysis.

## Steps

1. Load the Dataset
   The dataset is loaded using pandas:

```python
import pandas as pd

data = pd.read_csv('./data/Musical_instruments_reviews.csv')
data = data.head(1000)
```

1. Sentiment Analysis
   Using the VADER sentiment analysis tool, sentiment scores for each review are calculated and stored in a dictionary:

```python
from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
arr = {}
for i, row in data.iterrows():
    text = row['text']
    new_id = row['id']
    arr[new_id] = analyzer.polarity_scores(text)

vaders = pd.DataFrame(arr).T
vaders = vaders.reset_index().rename(columns={'index':'id'})
vaders = vaders.merge(data, how='left')
```

1. Visualize the Results
   Sentiment scores are visualized using seaborn bar plots:

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=vaders, x='score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()
```

## Files

- `Musical_instruments_reviews.csv`: The original dataset containing reviews of musical instruments.
- `vaders.csv`: The dataset with added sentiment scores for each review.
- `sentiment_analysis.py`: The script containing the sentiment analysis and visualization code.

## Usage

To run the project, execute the `sentiment_analysis.py` script. Ensure the dataset `Musical_instruments_reviews.csv` is in the correct directory (`./data/`).

## Conclusion

This project demonstrates how to perform sentiment analysis on a set of reviews and visualize the results using bar plots. The VADER sentiment analysis tool is used to calculate sentiment scores for each review, providing insights into the overall sentiment of the reviews.
