#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement 
# 
# With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.
# 
# As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 
# 
# In order to do this, you planned to build a sentiment-based product recommendation system

# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# #### Importing the packages

# In[ ]:


import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from xgboost import XGBClassifier
from sklearn.metrics.pairwise import pairwise_distances
import pickle
from nltk.corpus import stopwords
from datetime import datetime
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS

pd.set_option('display.max_colwidth', None)
pd.set_option('max_column', 100)

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# ##### Reading the data

# In[103]:


data_ebuss = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/capstone/sample30.csv')
data_ebuss.head()


# ##### Checking for any null values so as to perform data cleaning 

# In[ ]:


data_ebuss.isnull().sum()


# In[ ]:


data_ebuss['reviews_username'].head() # having a glimpse at the usernames


# ##### Having a look at data where usernames are present

# In[ ]:


data_ebuss[~data_ebuss['reviews_username'].isna()]


# ## EDA
# 
# Before we proceed any further it is highly important we understand how varied the data is across the whole dataset . We will perform EDA so as to understand trends in the data . In order to understand it better.

# ##### To understand user_sentiment across the data

# In[ ]:


sns.countplot(x='user_sentiment',data=data_ebuss)
plt.title('Distribution of user_sentiment in the dataset')
plt.show()


# ##### Takeaway :
# 
# Comparatively there are few negative reviews , which indicates class imbalance which should handled later during training of the model

# ##### To understand distribution of ratings

# In[ ]:


data_ebuss['reviews_rating'].value_counts()


# In[ ]:


sns.countplot(x='reviews_rating',data=data_ebuss)
plt.title('Distribution of ratings in the dataset')
plt.show()


# ##### Takeaway:
# Majority products are high-rated products which supports the dominance in the sentiment as well.

# ##### To understand word frequency and also to have a glimpse at what words are being used

# In this particular wordCloud we have used the data without filtering w.r.t to the sentiment. 

# In[104]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords,collocations=False).generate(' '.join(data_ebuss['reviews_text']))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In this particular wordcloud 'POSITIVE' user sentiment is considered.

# In[105]:


wordcloud = WordCloud(stopwords=stopwords,collocations=False).generate(' '.join(data_ebuss[data_ebuss['user_sentiment']=='Positive']['reviews_text']))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ##### Takeaway:
# We can see that a lot of positive adjectives are being used here.

# In this wordCloud negative user sentiment is being considered 

# In[106]:


wordcloud = WordCloud(stopwords=stopwords,collocations=False).generate(' '.join(data_ebuss[data_ebuss['user_sentiment']=='Negative']['reviews_text']))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ##### Takeaway:
# Upon looking keenly we can find few adjectives which align with the negative sentiment.

# In this wordClound we are only considering the title of the review

# In[107]:


wordcloud = WordCloud(stopwords=stopwords,collocations=False).generate(' '.join(data_ebuss['reviews_title'].fillna('')))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ##### Takeaway:
# There are few adjectives also majority are addressing to the product name

# ##### Considering Brand

# In[108]:


data_ebuss.brand.value_counts()


# Plotting for distribution of TOP-10 brands

# In[112]:


plt.figure(figsize=(15,15))
sns.countplot(data_ebuss['brand'], order=pd.value_counts(data_ebuss['brand']).iloc[:10].index)
plt.title('TOP 10 BRANDS')
plt.show()


# ##### Takeaway: 
# This shows the brand dominance by few brands on the market aka dataset.

# In[ ]:


data_ebuss.info() # having a glimpse at the data info


# In[ ]:


data_ebuss.shape #checking rows and cols


# In[ ]:


data_ebuss.nunique() #no of unique values present for respective columns 


# In[ ]:


data_ebuss.drop_duplicates().shape # checking to see if there are any duplicates


# No duplicates as of now 

# In[ ]:


data_ebuss = data_ebuss.drop_duplicates() # dropping the duplicates


# In[ ]:


help(pd.read_csv)


# ###### Reading the data description file for better understanding

# In[ ]:


desc_col = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/capstone/Data+Attribute+Description.csv',encoding='ISO-8859-1')


# In[ ]:


desc_col


# #### Understanding
# - If they would help the analysis how and why

# <table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>Attribute</th>      <th>Attribute Description</th>      <th>Would they be useful for analysis ?</th>      <th>Reason</th>    </tr>  </thead>  <tbody>    <tr>      <th>0</th>      <td>id</td>      <td>Uniques identity number to identify each unique review given by the user to a particular product in the dataset</td>      <td>True</td>      <td>Nothing</td>    </tr>    <tr>      <th>1</th>      <td>brand</td>      <td>Name of the brand of the product to which user has given review and rating</td>      <td>True</td>      <td>Helpful</td>    </tr>    <tr>      <th>2</th>      <td>categories</td>      <td>Category of the product like household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products, health care products and many more.</td>      <td>False</td>      <td>Would make things complex</td>    </tr>    <tr>      <th>3</th>      <td>manufacturer</td>      <td>Name of the manufacturer of the product</td>      <td>False</td>      <td>Not at all required</td>    </tr>    <tr>      <th>4</th>      <td>name</td>      <td>Name of the product to which user has added review or rating</td>      <td>True</td>      <td>Helpful</td>    </tr>    <tr>      <th>5</th>      <td>reviews_date</td>      <td>Date on which the review has been added by the user</td>      <td>False</td>      <td>Dates do not help us</td>    </tr>    <tr>      <th>6</th>      <td>reviews_didPurchase</td>      <td>Whether a particular user has purchased the product or not</td>      <td>False</td>      <td>Flags do not help us</td>    </tr>    <tr>      <th>7</th>      <td>reviews_doRecommend</td>      <td>Whether a particular user has recommended the product or not</td>      <td>False</td>      <td>Flags do not help us</td>    </tr>    <tr>      <th>8</th>      <td>reviews_rating</td>      <td>Rating given by the user to a particular product</td>      <td>True</td>      <td>Crucial for Analysis</td>    </tr>    <tr>      <th>9</th>      <td>reviews_text</td>      <td>Review given by the user to a particular product</td>      <td>True</td>      <td>Crucial for Analysis</td>    </tr>    <tr>      <th>10</th>      <td>reviews_title</td>      <td>The title of the review given by the user to a particular product</td>      <td>True</td>      <td>Crucial for Analysis</td>    </tr>    <tr>      <th>11</th>      <td>reviews_userCity</td>      <td>The residing city of the user</td>      <td>False</td>      <td>What we do has no dependency on location</td>    </tr>    <tr>      <th>12</th>      <td>reviews_userProvince</td>      <td>The residing province of the user</td>      <td>False</td>      <td>What we do has no dependency on location</td>    </tr>    <tr>      <th>13</th>      <td>reviews_username</td>      <td>The unique identification for individual user in the dataset</td>      <td>True</td>      <td>Crucial for Analysis</td>    </tr>    <tr>      <th>14</th>      <td>user_sentiment</td>      <td>The overall sentiment of the user for a particular product (Positive or Negative)</td>      <td>True</td>      <td>Target for the Sentiment Analysis</td>    </tr>  </tbody></table>

# In[ ]:


cols_unwanted = ['categories',
 'manufacturer',
 'reviews_date',
 'reviews_didPurchase',
 'reviews_doRecommend',
 'reviews_userCity',
 'reviews_userProvince']
# Defining unwanted columns taking into consideration the above reason


# In[ ]:


data_ebuss.drop(cols_unwanted, axis=1, inplace=True) # Dropping the unwanted columns
data_ebuss.shape


# In[ ]:


data_ebuss.drop_duplicates(inplace=True) # dropping duplicate rows now 


# In[ ]:


data_ebuss.shape # checking the shape


# No of duplicates : 279 Duplicates (30k - 29721)

# ##### Checking nulls for username

# In[ ]:


data_ebuss.isnull().sum() # checking null values


# In[ ]:


data_ebuss['reviews_username'].value_counts()


# In[ ]:


data_ebuss = data_ebuss[~data_ebuss.reviews_username.isna()] # dropping rows with no username


# Initially dropping rows with no username because they would create rather more confusion upon replacement and they would not be of much use without username in further analysis and also constitute to very less amount of percentage.

# In[ ]:


data_ebuss.shape # Checking the shape


# No of rows with no username : 29721-29658 = 63 

# ##### Moving over to user_sentiment

# In[ ]:


data_ebuss.isnull().sum() # checking no of null values


# Going ahead with the user sentiment

# In[ ]:


data_ebuss[data_ebuss['user_sentiment'].isna()] # looking at the data for the corresponding row


# This certainly looks like a positive review

# In[ ]:


data_ebuss['user_sentiment'].value_counts(dropna=False) # Checking value counts including null(NaN)


# In[ ]:


data_ebuss.isnull().sum()


# This looks like a positive review so changing it to Positive

# In[ ]:


data_ebuss.user_sentiment.fillna('Positive',inplace=True) # replacing it with Positive


# In[ ]:


data_ebuss.user_sentiment.value_counts()


# #### Cleaning up reviews title

# In[ ]:


data_ebuss.isnull().sum() # checking no of null values


# Replacing na with empty string as title would not effect much , so an empty string would be an apt solution.

# In[ ]:


data_ebuss.reviews_title.value_counts(dropna=False)


# In[ ]:


data_ebuss.reviews_title.fillna('',inplace=True) #replacing Nan with empty string


# In[ ]:


data_ebuss.reviews_title.value_counts(dropna=False) # checking value counts


# In[ ]:


data_ebuss.isnull().sum() # checking null values again


# As there are no null values , going forward to the other prospects of data cleaning.

# For processing purposes , to reduce load or weights being assigned we can combine reviews_title and reviews_text

# In[ ]:


data_ebuss['review_modified'] = data_ebuss['reviews_title']+' '+data_ebuss['reviews_text'] # Combining title & text


# In[ ]:


data_ebuss.head() 


# In[ ]:


data_ebuss.drop(['reviews_title','reviews_text'], axis=1, inplace=True) # dropping the other two as we have review_modified


# In[ ]:


data_ebuss.head()


# In[ ]:


data_ebuss.info() # checking rows and columns 


# ## TEXT PROCESSING

# In[ ]:


text_data_ebuss = data_ebuss.copy() # copying dataset for text processing


# ##### Converting them to lower case 

# In[ ]:


text_data_ebuss['review_modified'] = text_data_ebuss['review_modified'].apply(lambda x : str(x).lower())


# In[ ]:


text_data_ebuss.head()


# In[ ]:


text_data_ebuss.tail()


# ##### Removing special characters

# In[ ]:


sample_text = 'softness!! i got this conditioner with influenster to try it and im loving it so far, i have oily hair so i use it only in the ends of my hair and feels amazing, so soft and no mess!! this review was collected as part of a promotion.'
# testing out on a sample text


# In[ ]:


text_after_rem_punc = re.sub(r"[^A-Za-z0-9\s]",'',sample_text)
text_after_rem_punc # using regex sub method


# In[ ]:


s_text = '__ujjna hello 90 JEEll'
text_after_rem = re.sub(r"[^A-Za-z0-9\s]",'',s_text)
text_after_rem


# #### NOTE:
# not using \w because it has _ included in it 

# In[ ]:


text_data_ebuss['review_modified'] = text_data_ebuss['review_modified'].apply(lambda x: re.sub(r"[^A-Za-z0-9\s]",'',x))


# In[ ]:


text_data_ebuss.head()


# ##### Removing StopWords

# In[ ]:


stop_words = stopwords.words('english')
stop_words


# In[ ]:


sample_review = 'irritation my husband bought this gel for us the gel caused irritation and it felt like it was burning my skin i wouldnt recommend this gel'
# Trying it out on a sample review 


# In[ ]:


" ".join([word for word in sample_review.split() if word not in stop_words])


# In[ ]:


text_data_ebuss['review_modified'] = text_data_ebuss['review_modified'].apply(lambda x:" ".join([word for word in x.split() if word not in stop_words]))


# In[ ]:


text_data_ebuss.head()


# ##### Lemmatization

# In[ ]:


lemmatizer = WordNetLemmatizer()


# Before Lemma

# In[2]:


sample_before_lemma = 'disappointed read reviews looking buying one couples lubricants ultimately disappointed didnt even live reviews read starters neither boyfriend could notice sort enhanced captivating sensation notice however messy consistency reminiscent liquidy vaseline difficult clean pleasant especially since lacked captivating sensation expecting im disappointed paid much lube wont use could use normal personal lubricant 1 less money 2 less mess'
sample_before_lemma # trying it out on a sample


# After Lemma

# In[ ]:


' '.join([lemmatizer.lemmatize(word) for word in sample_before_lemma.split()])


# In[ ]:


text_data_ebuss['review_modified'] = text_data_ebuss['review_modified'].apply(lambda x:' '.join([lemmatizer.lemmatize(word) for word in x.split()]))


# In[ ]:


text_data_ebuss.head()


# In[ ]:


# Saving this in a pickle file for future purposes
now = datetime.now().strftime("%Y%m%d_%H%M")
pickle.dump(text_data_ebuss, open(f'TextProcessedData_{now}.pkl', 'wb'))
print("text_processed_df Saved as: TextProcessedData_{0}.pkl".format(now))


# #### EDA on Processed-Data

# Performing EDA on processed data to understand frequency on important words and their usage with regards to the sentiment.

# WordCLoud for review_modified where user_sentiment is Positive

# In[ ]:


# EDA before sentiment analysis
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords,collocations=False).generate(' '.join(text_data_ebuss[text_data_ebuss['user_sentiment']=='Positive']['review_modified']))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ##### Takeaway:
# More refined adjectives in their root form 

# WordCloud for review_modified where user_sentiment is Negative

# In[ ]:


wordcloud = WordCloud(stopwords=stopwords,collocations=False, max_font_size=30).generate(' '.join(text_data_ebuss[text_data_ebuss['user_sentiment']=='Negative']['review_modified']))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ##### Takeaway:
# More adjectives which indicate the sentiment

# ## Sentiment Analysis

# In[ ]:


sentiment_anal_cols = ['review_modified','user_sentiment'] # we need only two columns for this purpose


# In[ ]:


df_sentiment = text_data_ebuss[sentiment_anal_cols].copy() # copying required sub-set of text_data_ebuss
df_sentiment.head()


# ##### Mapping 

# In[ ]:


df_sentiment['user_sentiment'] = df_sentiment['user_sentiment'].map({'Positive':1,'Negative':0})
df_sentiment.head()


# ### Feature Extraction using tf-idf Vectorizer

# In[ ]:


help(TfidfVectorizer)


# In[ ]:


tfidf_vectorizer =  TfidfVectorizer(ngram_range=(1,2)) # includes word_length 1,2
X = tfidf_vectorizer.fit_transform(df_sentiment['review_modified'])
y = df_sentiment['user_sentiment']


# In[ ]:


# saving o/p from the vectorizer in a pickle file
now = datetime.now().strftime("%Y%m%d_%H%M")
pickle.dump(tfidf_vectorizer.vocabulary_, open(f'VectorizerFeatures_{now}.pkl', 'wb'))
print("vectorizer.vocabulary_ saved as: VectorizerFeatures_{0}.pkl".format(now))


# In[ ]:


# Splitting into train-test
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=30)
print(f' X-train shape --- {X_train.shape} \n y-train shape --- {y_train.shape} \n X-test shape --- {X_test.shape} \n y-test shape --- {y_test.shape}')


# In[ ]:


y_train.value_counts() # checking value_counts for user_sentiment


# There is very high class-imbalance which has to be handled

# ## Using SMOTE ( handling Class Imbalance)
# 
# - Perhaps the most widely used approach to synthesizing new examples is called the Synthetic Minority Oversampling TEchnique, or SMOTE for short.
# 
# - SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.
# 

# In[ ]:


smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)


# In[ ]:


y_train.value_counts() 


# SMOTE fixes our problem with Class Imbalance

# ##### Defining helper functions for model-building 

# In[ ]:


help(f1_score)


# In[ ]:


help(accuracy_score)


# In[ ]:


help(f1_score)


# In[ ]:


help(roc_auc_score)


# In[ ]:


evaluation_df = pd.DataFrame(columns=['model', 'Accuracy(train)','Accuracy(test)', 'Sensitivity(train)','Sensitivity(test)','Specificity(train)','Specificity(test)','F1 (train)', 'F1 (test)',
                                     'AUC (train)', 'AUC (test)'])
# defining data-frame with columns

def model_evaluation(model, X_train, X_test, y_train, y_test):
    """ Helper function to assess the performance of the given model with the respective data """
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    accuracy_train = accuracy_score(y_train, train_pred)
    accuracy_test = accuracy_score(y_test, test_pred)
    cm_train = confusion_matrix(y_train, train_pred)
    cm_test = confusion_matrix(y_test, test_pred)
    sensitivity_train = cm_train[1][1]/sum(cm_train[1])
    sensitivity_test = cm_test[1][1]/sum(cm_test[1])
    specificity_train = cm_train[0][0]/sum(cm_train[0])
    specificity_test = cm_test[0][0]/sum(cm_test[0])
    f1_score_train = f1_score(y_train, train_pred)
    f1_score_test =f1_score(y_test, test_pred)
    roc_train = roc_auc_score(y_train, train_pred)
    roc_test = roc_auc_score(y_test, test_pred)
    
    global evaluation_df
    evaluation_df = evaluation_df.append({'model':type(model),'Accuracy(train)':accuracy_train,'Accuracy(test)': accuracy_test, 'Sensitivity(train)':sensitivity_train, 'Sensitivity(test)':sensitivity_test, 'Specificity(train)':specificity_train
                         ,'Specificity(test)': specificity_test, 'F1 (train)': f1_score_train,  'F1 (test)':f1_score_test,'AUC (train)': roc_train, 'AUC (test)': roc_test} , ignore_index=True)
    
    


# ##### Model I : Logistic Regression with no hyperparameter tuning

# In[ ]:


lr = LogisticRegression(random_state=30)
lr.fit(X_train, y_train)


# In[ ]:


model_evaluation(lr, X_train, X_test, y_train, y_test)


# In[ ]:


evaluation_df


# ##### Takeaway :
# - All the params look good especially f1 and auc_roc 
# - But coming to test params Specificity doesn't shine which means it could be better at predicting whether a review is -ve.

# ##### Model II : XGBoost

# In[ ]:


help(XGBClassifier)


# In[ ]:


xg_boost = XGBClassifier(random_state=30)
xg_boost.fit(X_train, y_train)


# In[ ]:


model_evaluation(xg_boost, X_train, X_test, y_train, y_test)


# In[ ]:


evaluation_df.loc[1]


# ##### Takeaways: 
# - Neither bad nor great.
# - Only good thing is that there is very little difference in the train-test params except Specificity

# ##### Model III : Random Forest

# In[ ]:


help(RandomForestClassifier)


# In[ ]:


random_forest_classifier = RandomForestClassifier(random_state=30)
random_forest_classifier.fit(X_train, y_train)


# In[ ]:


model_evaluation(random_forest_classifier, X_train, X_test, y_train, y_test)


# In[ ]:


evaluation_df.loc[2]


# ##### Takeaways: 
# - Only good thing is that sensitivity is great.
# - But Specificity and AUC for test look very bad.

# ##### Model IV: Naive Bayes

# In[ ]:


help(MultinomialNB)


# In[ ]:


nb = MultinomialNB()
nb.fit(X_train, y_train)


# In[ ]:


model_evaluation(nb, X_train, X_test, y_train, y_test)


# In[ ]:


evaluation_df.loc[3]


# ##### Takeaways:
# - Everything looks mediocre.
# - Again not great specificity

# ##### Action:
# - Keeping in mind the results of all models , the specificity and auc (Test) of only LR is comparatively great.
# - Performing tuning on LR

# ##### Trying out the best model with Hyperparameter Tuning

# In[ ]:


log_model = LogisticRegression(random_state=30)
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet'],
    'C' : np.logspace(-2, 2, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    }
]


# In[ ]:


clf = GridSearchCV(log_model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1, scoring='roc_auc')


# In[ ]:


best_clf = clf.fit(X_train,y_train)


# In[ ]:


clf.best_params_


# In[ ]:


final_lr = LogisticRegression(penalty='l2',solver='liblinear',C=100,random_state=30)
final_lr.fit(X_train, y_train)


# In[ ]:


model_evaluation(final_lr, X_train, X_test, y_train, y_test) # training on the best found hyper-parameters


# In[ ]:


evaluation_df # looking at all the models and their scores collectively


# ##### Conclusion:
# - It clearly shows that LR is the winner.
# - Although other models seem to have high accuracy they fail at auc and specificity .

# In[ ]:


# Saving in a pickle file the first and best model LR
now = datetime.now().strftime("%Y%m%d_%H%M")
saved_model = pickle.dump(lr, open(f'LogisticRegressionModel_{now}.pkl', 'wb'))
print('Model Saved as: LogisticRegressionModel_{0}.pkl'.format(now))


# ## Product Recommendation Engine
# There are different parts for better discretion and understanding:
# - User based recommendation
# - User based prediction & evaluation
# - Item based recommendation
# - Item based prediction & evaluation

# #### Col selection:
# - Considering item aka id, user aka username , ratings

# In[96]:


recommendation_columns = ['id', 'reviews_username', 'reviews_rating'] # Considering these recommendation columns
df_recom = data_ebuss[recommendation_columns].copy()


# In[ ]:


df_recom.head() # having a glimpse at data


# In[ ]:


df_recom.shape


# In[ ]:


df_recom.isnull().sum() # check for null values


# #### Dividing the dataset into train and test

# In[ ]:


train_rat, test_rat = train_test_split(df_recom, train_size=0.8, random_state=30)
# Test and Train split of the dataset.


# In[ ]:


print(train_rat.shape)
print(test_rat.shape)


# In[ ]:


# Pivot the train ratings' dataset into matrix format in which columns item-ids and the rows are usernames.
df_pivot = train_rat.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating',fill_value=0
).fillna(0)

df_pivot.head(3)


# #### Creating dummy train & dummy test dataset
# These dataset will be used for prediction 
# - Dummy train will be used later for prediction of the items which has not been rated by the user. To ignore the movies rated by the user, we will mark it as 0 during prediction. The items not rated by user is marked as 1 for prediction in dummy train dataset. 
# 
# - Dummy test will be used for evaluation. To evaluate, we will only make prediction on the items rated by the user. So, this is marked as 1. This is just opposite of dummy_train.

# In[ ]:


# Copy the train dataset into dummy_train
dummy_train = train_rat.copy()


# In[ ]:


dummy_train['reviews_rating'].describe() # Checking min-value for the rating . 


# In[ ]:


# The movies not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[ ]:


# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating',
    fill_value=1
).fillna(1)


# In[ ]:


dummy_train.head()


# ### Using Cosine Similarity ( User Similarity Matrix)

# **Cosine Similarity**
# 
# Cosine Similarity is a measurement that quantifies the similarity between two vectors [Which is Rating Vector in this case] 
# 
# **Adjusted Cosine**
# 
# Adjusted cosine similarity is a modified version of vector-based similarity where we incorporate the fact that different users have different ratings schemes. In other words, some users might rate items highly in general, and others might give items lower ratings as a preference. To handle this nature from rating given by user , we subtract average ratings for each user from each user's rating for different items.
# 
# 

# In[ ]:


# Create a user-item matrix.
df_pivot = train_rat.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
)

df_pivot.head()


# In[ ]:


# Normalising the rating of the movie for each user around 0 mean
mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[ ]:


df_subtracted.head()


# In[ ]:


# Creating the User Similarity Matrix using pairwise_distance function.
# Finding cosine similarity
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[ ]:


user_correlation.shape


# ### Prediction - User User
# 

# Doing the prediction for the users which are positively related with other users, and not the users which are negatively related as we are interested in the users which are more similar to the current users. So, ignoring the correlation for values less than 0. 

# In[ ]:


user_correlation[user_correlation<0]=0
user_correlation


# Rating predicted by the user (for movies rated as well as not rated) is the weighted sum of correlation with the movie rating (as present in the rating dataset). 

# In[ ]:


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings


# In[ ]:


user_predicted_ratings.shape


# Since we are interested only in the items not rated by the user, we will ignore the items rated by the user by making it zero. 

# In[ ]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# ### Evaluation - User User 
# 
# Evaluation will we same as you have seen above for the prediction. The only difference being, you will evaluate for the movie already rated by the user insead of predicting it for the item not rated by the user. 

# In[ ]:


common = test_rat[test_rat.reviews_username.isin(train_rat.reviews_username)]
common.shape # getting the common usernames in train and test


# In[ ]:


common.head()


# In[ ]:


# convert this into user-item matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating')
# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)

user_correlation_df['reviews_username'] = df_subtracted.index
user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()


# In[ ]:


list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_subtracted.index.tolist()

user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)] # checking if present

user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)] 

user_correlation_df_3 = user_correlation_df_2.T

user_correlation_df_3.head()


# In[ ]:


user_correlation_df_3.shape


# In[ ]:


user_correlation_df_3[user_correlation_df_3<0]=0
common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings


# In[ ]:


dummy_test = common.copy()
dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_test = dummy_test.pivot_table(index='reviews_username', columns='id', values='reviews_rating').fillna(0)
dummy_test.shape


# In[ ]:


common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)
common_user_predicted_ratings.head(2) 


# In[ ]:


X  = common_user_predicted_ratings.copy() 
X = X[X>0]
# Calculating the RMSE for only the items rated by user. For RMSE, normalising the rating to (1,5) range.
scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))
print(y)


# In[ ]:


common_ = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating')
# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))
rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# #### Takeaway :
# - The rmse looks mediocre let's go ahead with Item based similarity.

# ## Item Based Similarity

# Taking the transpose of the rating matrix to normalize the rating around the mean for different item ID. In the user based similarity, we had taken mean for each user instead of each item. 

# In[ ]:


df_pivot = train_rat.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).T

df_pivot.head()


# Normalising the movie rating for each movie for using the Adjusted Cosine

# In[ ]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T

df_subtracted.head()


# Finding the cosine similarity using pairwise distances approach

# In[ ]:


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# In[ ]:


item_correlation[item_correlation<0]=0
item_correlation


# ### Prediction - Item Item

# In[ ]:


item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings


# In[ ]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()
# Filtering the rating only for the movies not rated by the user for recommendation


# ### Evaluation - Item Item

# Evaluation will we same as you have seen above for the prediction. The only difference being, you will evaluate for the item aka product already rated by the user instead of predicting it for the item not rated by the user. 

# In[ ]:


common =  test_rat[test_rat.id.isin(train_rat.id)]
common.shape


# In[ ]:


common.head(4)


# In[ ]:


common_item_based_matrix = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T
common_item_based_matrix.shape


# In[ ]:


item_correlation_df = pd.DataFrame(item_correlation)
item_correlation_df.head()


# In[ ]:


item_correlation_df['id'] = df_subtracted.index
item_correlation_df.set_index('id',inplace=True)
item_correlation_df.head()


# In[ ]:


list_name = common.id.tolist()

item_correlation_df.columns = df_subtracted.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]

item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T

item_correlation_df_3.head()


# In[ ]:


item_correlation_df_3[item_correlation_df_3<0]=0

common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings


# In[ ]:


common_item_predicted_ratings.shape


# Dummy test will be used for evaluation. To evaluate, we will only make prediction on the movies rated by the user. So, this is marked as 1. This is just opposite of dummy_train

# In[ ]:


dummy_test = common.copy()
dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_test = dummy_test.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T.fillna(0)
# The products not rated is marked as 0 for evaluation. And make the item- item matrix representaion.

common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)
common_ = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[ ]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# ##### Takeaway : 
# - RMSE is comparatively higher than User based.

# #### Conclusion :
# - Going ahead with user-based similarity

# In[97]:


# Copy the train dataset into dummy_
dummy_ = df_recom.copy()

# The products not rated by user is marked as 1 for prediction. 
dummy_['reviews_rating'] = dummy_['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

# # Convert the dummy train dataset into matrix format.
dummy_ = dummy_.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating',
    fill_value=1
)

# # Create a user-product matrix.
df_pivot = df_recom.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
)

mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T

# # Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
user_correlation[user_correlation<0]=0
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_final_rating = np.multiply(user_predicted_ratings, dummy_)


# In[ ]:


# Dumping it in a pickle file
now = datetime.now().strftime("%Y%m%d_%H%M")
pickle.dump(user_final_rating, open(f'user_final_rating_iter_{now}.pkl', 'wb'))
print(f'text_processed_df Saved as: user_final_rating_{0}.pkl'.format(now))


# In[83]:


# paths of all the pickle files saved
pro = '/content/gdrive/MyDrive/Colab Notebooks/capstone/TextProcessedData_20220322_2004.pkl'
rating = "/content/gdrive/MyDrive/Colab Notebooks/capstone/user_final_rating_iter_20220322_2104.pkl"
model = '/content/gdrive/MyDrive/Colab Notebooks/capstone/LogisticRegressionModel_20220322_2006.pkl'
features = '/content/gdrive/MyDrive/Colab Notebooks/capstone/VectorizerFeatures_20220322_2007.pkl'


# In[85]:


pro_data = pickle.load(open(pro,'rb'))
ratings = pickle.load(open(rating,'rb'))
model_pred = pickle.load(open(model,'rb'))
features = pickle.load(open(features, 'rb'))


# #### Helper functions for predicting top 5 products for users

# In[114]:


def get_top20_prod_based_on_recommendation(name):
    name = name.lower()
    user_final = user_final_rating.loc[name]
    user_final = user_final.sort_values(ascending=False)[:20]
    user_final = user_final.index.to_list()
    return user_final
 


# In[100]:


def get_top5_prod_based_on_sentiment_of_reviews(top_20_prod):
        
    reviews_top_20 = pro_data.loc[pro_data.id.isin(top_20_prod), ['id', 'name', 'review_modified']]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), vocabulary=features)
    temp = vectorizer.fit_transform(reviews_top_20['review_modified'])
    reviews_top_20['predicted'] = model_pred.predict(temp)

    df = reviews_top_20.groupby('id',as_index=False)
    df = df.agg({'name':'max', 'predicted': ['count','sum']})
    df.columns = ['id', 'name', 'pred_total', 'pred_pos']
    
    df['positive_perc_review'] = round(df.pred_pos*100/df.pred_total, 2)
    df.sort_values('positive_perc_review', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df.loc[:4, ['name', 'positive_perc_review']]


# In[88]:


def get_recommendation_from_username(name):
    top20 = get_top20_prod_based_on_recommendation(name)
    top5 = get_top5_prod_based_on_sentiment_of_reviews(top20)
    return top5


# In[115]:


get_recommendation_from_username('mike')


# In[102]:


get_recommendation_from_username('linda')

