import pandas as pd
import random
from selenium.webdriver import (Chrome)
import pickle
import time
import sys

def scrape_data(list_name, id, url):
    documents = []
    browser = Chrome()
    contains_next_page = True
    while (contains_next_page and len(documents) < 250):
        print ('Inside While:',len(documents), url)
        time.sleep(random.choice([20,25,30,35,40,45]))
        browser.get(url)
        html = browser.page_source
        documents.append(html)
        print ('Before Try:',len(documents))
        try:
            search_button = browser.find_element_by_css_selector("li.a-last")
            links = search_button.find_element_by_css_selector('a')
            url = links.get_attribute("href")
            print (url)
        except:
            contains_next_page = False
            print('failed')
    if (len(documents) > 0):
        pickle.dump( documents, open( f'data/{list_name}_{id}', "wb" ) )
        return True
    else:
        return False

def load_scrape_list(filename):
    df = pd.read_csv(filename)
    for row in df[df['Status'] == 1].iterrows():
       id = row[1][0]
       url = row[1][4]
       scrape_data('scrape_list',id, url)
       df['Status'][row[0]] = 2

list_name = sys.argv[1]
id = sys.argv[2]
url = sys.argv[3]
scrape_data(list_name, id, url)





