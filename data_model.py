from constants import *
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


class ProductRecommendation():
    """ Class for handling product recommendation"""

    def valid_input_checker(self, name):
        """ To check for valid username"""
        df = pd.read_csv(RAW_DATA_PATH)
        valid_usernames = df.reviews_username.str.lower().unique().tolist()
        return name.lower() in valid_usernames

    def get_top20_products(self, name):
        """ To get top 20 products for the recommendation system"""
        name = name.lower()
        ratings = pickle.load(open(RATINGS_PATH, 'rb'))
        ratings = ratings.loc[name]
        ratings = ratings.sort_values(ascending=False)[:20]
        ratings = ratings.index.to_list()
        return ratings 

    def get_top5_products(self, top_20_prod):
        """ Using the best picked model , the tf-idf features for the top 5"""

        processed_data = pickle.load(open(PROCESSED_DATA_PATH,'rb'))
        features = pickle.load(open(FEATURES_PATH, 'rb'))
        model_best = pickle.load(open(MODEL_BEST_PATH,'rb'))

        # filtering out the id's and using required col names
        reviews_top_20 = processed_data.loc[processed_data.id.isin(top_20_prod), ['id', 'brand', 'name', 'review_modified']]
        
        vectorizer = TfidfVectorizer(ngram_range=(1,2), vocabulary=features)
        vectorized_input = vectorizer.fit_transform(reviews_top_20['review_modified'])

        # using the model to predict
        reviews_top_20['predicted'] = model_best.predict(vectorized_input)

        df = reviews_top_20.groupby('id',as_index=False)
        df = df.agg({'name':'max', 'brand': 'max', 'predicted': ['count','sum']})
        df.columns = ['id', 'name', 'brand', 'pred_total', 'pred_pos']
        
        # calculating the percentage
        df['positive_perc_review'] = round(df.pred_pos*100/df.pred_total, 2)
        df.sort_values('positive_perc_review', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # returning top 5 
        return df.loc[:4,['name', 'brand', 'positive_perc_review']]

    def get_recommendation_from_username(self, name):
        """ Binding function for recommender and sentiment analysis"""
        top20_users = self.get_top20_products(name)
        top5_products = self.get_top5_products(top20_users)
        return top5_products
