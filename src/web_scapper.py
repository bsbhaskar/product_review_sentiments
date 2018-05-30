import pandas as pd
import random
from selenium.webdriver import (Chrome)
import pickle
import time
import sys

'''
web scrapper to scrape amazon, bestbuy and walmart customer review links. Input to the file is a url to customer review page along with the site. The scrapper has a page clicker that collects all review pages for the given product and stores as a collection of html documents in file for future processing.
Future enhancements will include, idenfying customer review pages on major ecommerce sites, given a product identifier .
'''

def scrape_data(list_name, id, url, scraping_site='amazon'):
    '''
    method to scape webpage given url and site named.
    '''
    documents = []
    if (scraping_site == 'amazon'):
        next_page_class = 'li.a-last'
    elif (scraping_site == 'bestbuy'):
        next_page_class = "li.next"
    elif (scraping_site == 'walmart'):
        next_page_class = "button.paginator-btn-next"

    browser = Chrome()
    browser.get(url)
    contains_next_page = True

    while (contains_next_page and len(documents) < 250):
        time.sleep(random.choice(range(20,50)))
        try:
            html = browser.page_source
            documents.append(html)
            search_button = browser.find_element_by_css_selector(next_page_class)
            print (search_button)
            if (scraping_site == 'walmart'):
                search_button.click()
            else:
                links = search_button.find_element_by_css_selector('a')
                url = links.get_attribute("href")
                browser.get(url)
        except:
            contains_next_page = False
            print('failed')
    if (len(documents) > 0):
        pickle.dump( documents, open( f'data/scraped_data/{list_name}_{scraping_site}_{id}', "wb" ) )
        return True
    else:
        return False

def load_scrape_list(filename):
    '''
    load url data from a csv file
    '''
    df = pd.read_csv(filename)
    for row in df[df['Status'] == 1].iterrows():
       id = row[1][0]
       url = row[1][4]
       scrape_data('scrape_list',id, url)
       df['Status'][row[0]] = 2

list_name = sys.argv[1]
id = sys.argv[2]
url = sys.argv[3]
site = sys.argv[4]
scrape_data(list_name, id, url, site)
