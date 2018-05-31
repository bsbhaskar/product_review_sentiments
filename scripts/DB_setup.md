#Set up Postgres on aws
sudo apt install postgresql
start Postgres ( if not started ) - sudo service start postgresql
Login to Postgres:
	sudo su postgres
	psql
Create User
	ALTER ROLE {user_name} with LOGIN PASSWORD {user_password};
Create Database
	CREATE DATABASE {database_name};
Install psycopg2 so you can connect using python
	conda install pyycopg2

#Create DB Configfile in home directory
cd ~
mkdir .product_reviews
create a file named param.cfg with following. Replace {db_name}, {db_user} and {db_pwd} with the correct database name, username and password.

[DB]
db_name = {db_name}
db_user = {db_user}
db_pwd = {db_pwd}
db_host = localhost
