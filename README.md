# Product Insights Based On Customer Reviews
Analyzes Consumer Reviews to gain actionable insights on any given Product. The project uses three models - Multinomial Naïve Bayes Classifier, Latent Dirichlet Allocation and Word2Vec to identify key features within positive and negative reviews, cluster them into topics and find relationship between features, respectively. Selenium and Beautiful Soup is used to scrape and parse reviews and Natural Language Processing Pipeline vectorizes raw scraped text into TFIDF for analysis. Spacy and Gensim is used identify entities, phrases and sentiments.

Product owners interested in understanding what their customers are saying about their product can use this codebase to scrape online e-commerce sites, vectorize words and highlight key features.

Visit d3shopping.com for a demo of the product.

#Setting up Repository
After pulling down github repository, implement the following steps to get started.

1) set up Postgres Database. Instruction for setting Postgres DB are included in script/DB_setup file
2) Create reviews table using load sample data from data/sample_reviews.X_descr_vectors
3) Set-up Flask and start the application by calling 'python app.py' from the src directory.
