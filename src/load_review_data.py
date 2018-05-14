import os
import psycopg2
import configparser
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class ReviewDataLoader:

    def __init__(self):

        config = configparser.ConfigParser()

        config.read(os.path.expanduser('~/.product_reviews/param.cfg'))
        self.db_name = config['DB']['db_name']
        self.db_user = config['DB']['db_user']
        self.db_pwd = config['DB']['db_pwd']
        self.db_host = config['DB']['db_host']


    def retrieve_reviews(self,product):

        conn = psycopg2.connect(dbname=self.db_name, user=self.db_user, password=self.db_pwd, host=self.db_host)
        cursor = conn.cursor()
        cursor.execute("select * from reviews where r_comments like 'Verified%' and product_name = '{}'".format(product))
        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        columns = ['id','product','url','p_no','r_no','r_stars','r_date','r_name','r_title','r_text','r_comments']
        df.columns = columns
        cursor.close()
        conn.close()
        return df

    def retrieve_all_reviews(self):

        conn = psycopg2.connect(dbname=self.db_name, user=self.db_user, password=self.db_pwd, host=self.db_host)
        cursor = conn.cursor()
        cursor.execute("select * from reviews where r_comments like 'Verified%' ")
        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        columns = ['id','product','url','p_no','r_no','r_stars','r_date','r_name','r_title','r_text','r_comments']
        df.columns = columns
        cursor.close()
        conn.close()
        return df
