'''
This file contains methods to pull all rows or product specific rows into a dataframe.
'''
import os
import psycopg2
import configparser
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class ReviewDataLoader:

    def __init__(self):
        '''
        DB credentials are stored in param.cfg file in users root directory
        '''
        config = configparser.ConfigParser()
        config.read(os.path.expanduser('~/.product_reviews/param.cfg'))
        self.db_name = config['DB']['db_name']
        self.db_user = config['DB']['db_user']
        self.db_pwd = config['DB']['db_pwd']
        self.db_host = config['DB']['db_host']

    def parse_stars(self,r_stars):
        '''
        converts no of stars from amazon reviews into numerical ratings. In the future, this will happen in webparsing function
        '''
        rating = 0
        if (r_stars == '5.0 out of 5 stars'):
            rating = 5
        elif (r_stars == '4.0 out of 5 stars'):
            rating = 4
        elif (r_stars == '3.0 out of 5 stars'):
            rating = 3
        elif (r_stars == '2.0 out of 5 stars'):
            rating = 2
        elif (r_stars == '1.0 out of 5 stars'):
            rating = 1
        return rating

    def retrieve_reviews(self,product):
        '''
        retrives data for specific product from db
        '''
        conn = psycopg2.connect(dbname=self.db_name, user=self.db_user, password=self.db_pwd, host=self.db_host)
        cursor = conn.cursor()
        cursor.execute("select * from reviews where r_comments like 'Verified%' and model = '{}'".format(product))
        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        columns = ['id','product','url','p_no','r_no','r_stars','r_date','r_name','r_title','r_text','r_comments','brand_name','category','model']
        df.columns = columns
        df_new = df.copy()
        df_new['rating'] = df['r_stars'].apply(lambda x: self.parse_stars(x))
        df_new['reviews'] = df['r_title'] + ' ' + df['r_text']
        cursor.close()
        conn.close()
        return df_new

    def retrieve_all_reviews(self):
        '''
        retrives data for all products from database. make sure there is sufficient memory to handle the large data-set.
        '''

        conn = psycopg2.connect(dbname=self.db_name, user=self.db_user, password=self.db_pwd, host=self.db_host)
        cursor = conn.cursor()
        cursor.execute("select * from reviews where r_comments like 'Verified%' ")
        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        columns = ['id','product','url','p_no','r_no','r_stars','r_date','r_name','r_title','r_text','r_comments','brand_name','category','model']
        df.columns = columns
        df_new = df.copy()
        df_new['rating'] = df['r_stars'].apply(lambda x: self.parse_stars(x))
        df_new['reviews'] = df['r_title'] + ' ' + df['r_text']
        cursor.close()
        conn.close()
        return df_new
