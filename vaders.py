import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer


# Load the dataset
data = pd.read_csv('./data/Musical_instruments_reviews.csv')
data = data.head(1000)
analyzer = SentimentIntensityAnalyzer()
arr = {}
# Iterate through the DataFrame
for i, row in data.iterrows():
    text = row['text']
    new_id = row['id']
    arr[new_id] = analyzer.polarity_scores(text)

vaders = pd.DataFrame(arr).T
vaders = vaders.reset_index().rename(columns={'index':'id'})
vaders = vaders.merge(data,how='left')

fig,axs = plt.subplots(1,3,figsize=(15,5))
sns.barplot(data=vaders,x='score',y='pos',ax=axs[0])
sns.barplot(data=vaders,x='score',y='neu',ax=axs[1])
sns.barplot(data=vaders,x='score',y='neg',ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()