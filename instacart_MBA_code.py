# -*- coding: utf-8 -*-

## 1.1 Import the required packages
##The garbage collector (package gc), attempts to reclaim garbage, or memory occupied by objects (e.g., DataFrames) that are no longer in use by Python ([ref1](https://www.techopedia.com/definition/1083/garbage-collection-gc-general-programming), [ref2](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)). This package will eliminate our risk to exceed the 16GB threshold of available RAM that Kaggle offers.

##The **"as"** reserved word is to define an alias to the package. The alias help us to call easier a package in our code.


# For data manipulation
import pandas as pd         

# Garbage Collector to free up memory
import gc                         
gc.enable()                       # Activate

##since we ran the code on Google colabs we need to mount the files from Google drive
from google.colab import drive
drive.mount('/content/drive')

##we set the path for file 
path = "/content/drive/My Drive/DM_proj/DM_proj/DM_Project_code/"

"""## 1.2 Load data from the CSV files
Instacart provides 6 CSV files, which we have to load into Python. Towards this end, we use the .read_csv() function, which is included in the Pandas package. Reading in data with the .read_csv( ) function returns a DataFrame.
"""

orders = pd.read_csv(path + 'orders.csv' )
order_products_train = pd.read_csv(path + 'order_products__train.csv')
order_products_prior = pd.read_csv(path + 'order_products__prior.csv')
products = pd.read_csv(path + 'products.csv')
aisles = pd.read_csv(path + 'aisles.csv')
departments = pd.read_csv(path + 'departments.csv')

"""## 1.4 Create a DataFrame with the orders and the products that have been purchased on prior orders 
"""

#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner') ##
op.head()


# 2. Create Predictor Variables
#We are now ready to identify and calculate predictor variables based on the provided data. We can create various types of predictors such as:
#User predictors-describing the behavior of a user e.g. total number of orders of a user.
#Product predictors-describing characteristics of a product e.g. total number of times a product has been purchased.
#User & product predictors-describing the behavior of a user towards a specific product e.g. total times a user ordered a specific product.

## 2.1 Create user predictors
#We create the following predictors:
#Number of orders per customer
#How frequent a customer has reordered products

### 2.1.1 Number of orders per customer
#We calculate the total number of placed orders per customer. We create a **user** DataFrame to store the results.

## First approach in one step:
# Create distinct groups for each user, identify the highest order number in each group, save the new column to a DataFrame
user = op.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
user.head()

# Reset the index of the DF so to bring user_id from index to column (pre-requisite for step 2.4)
user = user.reset_index()
user.head()

### 2.1.2 How frequent a customer has reordered products
#This feature is a ratio which shows for each user in what extent has products that have been reordered in the past

u_reorder = op.groupby('user_id')['reordered'].mean().to_frame('u_reordered_ratio') #
u_reorder = u_reorder.reset_index()
u_reorder.head()

#The new feature will be merged with the user DataFrame (section 2.1.1) which keep all the features based on users. We perform a left join as we want to keep all the users that we have created on the user DataFrame

user = user.merge(u_reorder, on='user_id', how='left') #

del u_reorder
gc.collect()

user.head()

## 2.2 Create product predictors
#We create the following predictors:
#Number of purchases for each product
#What is the probability for a product to be reordered

### 2.2.1 Number of purchases for each product
#We calculate the total number of purchases for each product (from all customers). We create a **prd** DataFrame to store the results.

# Create distinct groups for each product, count the orders, save the result for each product to a new DataFrame  
prd = op.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')
prd.head()

# Reset the index of the DF so to bring product_id rom index to column (pre-requisite for step 2.4)
prd = prd.reset_index()
prd.head()

## 2.2.2 What is the probability for a product to be reordered

# execution time: 25 sec
# the x on lambda function is a temporary variable which represents each group
# shape[0] on a DataFrame returns the number of rows
p_reorder = op.groupby('product_id').filter(lambda x: x.shape[0] >40)
p_reorder.head()

### 2.2.2.2 Group products, calculate the mean of reorders
#To calculate the reorder probability we will use the aggregation function mean() to the reordered column. In the reorder data frame, the reordered column indicates that a product has been reordered when the value is 1.
#The .mean() calculates how many times a product has been reordered, divided by how many times has been ordered in total. 

p_reorder = p_reorder.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio')
p_reorder = p_reorder.reset_index()
p_reorder.head()

### 2.2.2.3 Merge the new feature on prd DataFrame
#Merge the prd DataFrame with reorder
prd = prd.merge(p_reorder, on='product_id', how='left')

#delete the reorder DataFrame
del p_reorder
gc.collect()

prd.head()

#Fill NaN values
#As you may notice, there are product with NaN values. This regards the products that have been purchased less than 40 times from all users and were not included in the p_reorder DataFrame. **As we performed a left join with prd DataFrame, all the rows with products that had less than 40 purchases from all users, will get a NaN value.**
#For these products we their NaN value with zero (0):

prd['p_reorder_ratio'] = prd['p_reorder_ratio'].fillna(0) #
prd.head()

#Our final DataFrame should not have any NaN values, otherwise the fitting processwill throw an error!

## 2.3 Create user-product predictors
#We create the following predictors:
#How many times a user bought a product
#How frequently a customer bought a product after its first purchase ?

### 2.3.1 How many times a user bought a product
#We create different groups that contain all the rows for each combination of user and product. With the aggregation function .count( ) we get how many times each user bought a product. We save the results on new **uxp** DataFrame.

# Create distinct groups for each combination of user and product, count orders, save the result for each user X product to a new DataFrame 
uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')
uxp.head()

# Reset the index of the DF so to bring user_id & product_id rom indices to columns (pre-requisite for step 2.4)
uxp = uxp.reset_index()
uxp.head()

### 2.3.2.1 Calculating the numerator - How many times a customer bought a product? ('Times_Bought_N')
times = op.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()

# 2.3.2.2 Calculating the denumerator
#To calculate the denumerator, we have first to calculate the total orders of each user & first order number for each user and every product purchase
#### 2.3.2.2.a The total number of orders for each customer ('total_orders'
#Here we .groupby( ) only by the user_id, we keep the column order_number and we get its highest value with the aggregation function .mean()

total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders') #
total_orders.head()

##The order number where the customer bought a product for first time ('first_order_number')
#Where for first_order_number we .groupby( ) by both user_id & product_id. As we want to get the order when a product has been purchases for first time, we select the order_number column and we retrieve with .min( ) aggregation function, the earliest order.


first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()

#We merge the first order number with the total_orders DataFrame. As total_orders refers to all users, where first_order_no refers to unique combinations of user & product, we perform a right join:

span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()

#For each product get the total orders placed since its first order ('Order_Range_D')
#The denominator now can be created with simple operations between the columns of results DataFrame:
# The +1 includes in the difference the first order were the product has been purchased
span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()

#Create the final ratio "uxp_reorder_ratio"
#### 2.3.2.3.a Merge the DataFrames of numerator & denumerator
#We select to merge **times** DataFrame which contains the numerator & **span** which contains the denumerator of our desired ratio. **As both variables derived from the combination of users & products, any type of join will keep all the combinations.**

uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()

#Perform the final division #
#Here we divide the Times_Bought_N by the Order_Range_D for each user and product.

uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D ##
uxp_ratio.head()

#Keep the final feature
We select to keep only the 'user_id', 'product_id' and the final feature 'uxp_reorder_ratio'

uxp_ratio = uxp_ratio.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D'], axis=1)
uxp_ratio.head()

#Remove temporary DataFrames
del [times, first_order_no, span]

#Merge the final feature with uxp DataFrame
#The new feature will be merged with the uxp DataFrame (section 2.3.1) which keep all the features based on combinations of user-products. We perform a left join as we want to keep all the user-products that we have created on the uxp DataFrame

uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')

del uxp_ratio
uxp.head()

#Merge all features
#We now merge the DataFrames with the three types of predictors that we have created (i.e., for the users, the products and the combinations of users and products).


#Merge uxp with user DataFrame
#Here we select to perform a left join of uxp with user DataFrame based on matching key "user_id"

#Merge uxp features with the user features
#Store the results on a new DataFrame
data = uxp.merge(user, on='user_id', how='left')
data.head()

#Merge data with prd DataFrame
#In this step we continue with our new DataFrame **data** and we perform a left join with prd DataFrame. The matching key here is the "product_id".

#Merge uxp & user features (the new DataFrame) with prd features
data = data.merge(prd, on='product_id', how='left')
data.head()

#Delete previous DataFrames
#The information from the DataFrames that we have created to store our features (op, user, prd, uxp) is now stored on **data**. 
#As we won't use them anymore, we now delete them.

del op, user, prd, uxp
gc.collect()


################################################
################################################
################################################
################################################

#3 Create train and test DataFrames
## 3.1 Include information about the last order of each user

# In two steps keep only the future orders from all customers: train & test 
orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head(10)

# bring the info of the future orders to data DF
data = data.merge(orders_future, on='user_id', how='left')
data.head(10)

# 3.2 Prepare the train DataFrame
#In order to prepare the train Dataset, which will be used to create our prediction model, we need to include also the response (Y) and thus have the following structure:

#Keep only the customers who we know what they bought in their future order
data_train = data[data.eval_set=='train']
data_train.head()


#Get from order_products_train all the products that the train users bought bought in their future order
data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head(15)

"""On the last columm (reordered) you can find out our response (y). 
There are combinations of User X Product which they were reordered (1) on last order where other were not (NaN value).

Now we manipulate the data_train DataFrame, to bring it into a structure for Machine Learning (X1,X2,....,Xn, y):
- Fill NaN values with value zero (regards reordered rows without value = 1)
"""

#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.
data_train['reordered'] = data_train['reordered'].fillna(0)
data_train.head(15)

"""- Set as index the column(s) that describe uniquely each row (in our case "user_id" & "product_id")"""

#We set user_id and product_id as the index of the DF
data_train = data_train.set_index(['user_id', 'product_id'])
data_train.head(15)

"""- Remove columns which are not predictors (in our case: 'eval_set','order_id')"""

#We remove all non-predictor variables
data_train = data_train.drop(['eval_set', 'order_id'], axis=1)
data_train.head(15)

"""## 3.3 Prepare the test DataFrame
The test DataFrame must have the same structure as the train DataFrame, excluding the "reordered" column (as it is the label that we want to predict).
<img style="float: left;" src="https://i.imgur.com/lLJ7wpA.jpg" >

 To create it, we:
- Keep only the customers who are labelled as test
"""

#Keep only the future orders from customers who are labelled as test
data_test = data[data.eval_set=='test']
data_test.head()

"""- Set as index the column(s) that uniquely describe each row (in our case "user_id" & "product_id")"""

#We set user_id and product_id as the index of the DF
data_test = data_test.set_index(['user_id', 'product_id'])
data_test.head()

"""- Remove the columns that are predictors (in our case:'eval_set', 'order_id')"""

#We remove all non-predictor variables
data_test = data_test.drop(['eval_set','order_id'], axis=1)
#Check if the data_test DF, has the same number of columns as the data_train DF, excluding the response variable
data_test.head()


################################################
################################################################################################
################################################################################################
################################################################################################


# 4. Create predictive model (fit)
##The Machine Learning model that we are going to create is based on the Random Forest Algorithm.


############################### RANDOM FOREST ###############################################


# TRAIN FULL
## IMPORT REQUIRED PACKAGES
from sklearn.ensemble import RandomForestClassifier

## SPLIT DF TO: X_train, y_train (axis=1)
X_train, y_train = data_train.drop('reordered', axis=1), data_train.reordered

## INITIATE AND TRAIN MODEL
rfc = RandomForestClassifier(n_estimators=10, n_jobs=-1 ,random_state=42)
model = rfc.fit(X_train, y_train)


# TRAIN 80% - VALIDATE 20% 


##IMPORT REQUIRED PACKAGES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split #validate algorithm
from sklearn.metrics import f1_score, classification_report, confusion_matrix


## SPLIT DF TO: 80% for training and 20% as validation (axis=0)& THEN TO to X_train, X_val, y_train, y_val (axis=1)

X_train, X_val, y_train, y_val = train_test_split(data_train.drop('reordered', axis=1), data_train.reordered, test_size=0.8, random_state=42)


## INITIATE AND TRAIN MODEL

rfc = RandomForestClassifier(n_estimators=10, n_jobs=-1 ,random_state=42)
model = rfc.fit(X_train, y_train) #


## SCORE MODEL ON VALIDATION SET

### Predict on validation set with fixed threshold

y_val_pred = (model.predict_proba(X_val)[:,1] >= 0.30).astype(int)

### Get scores on validation set
print("RESULTS ON VALIDATION SET\n====================")
print("F1 Score: ",f1_score(y_val, y_val_pred, average='binary'), "\n====================")
print("Classification Report\n ", classification_report(y_val, y_val_pred), "\n====================")
print("Confusion Matrix\n ", confusion_matrix(y_val, y_val_pred))

### Remove validate algorithm objects
del [X_val, y_val]


# FEATURE IMPORTANCE - AS DF
feature_importances_df = pd.DataFrame(model.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances_df)


# FEATURE IMPORTANCE - GRAPHICAL
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
feat_importances.plot(kind='barh')


# DELETE TEMPORARY OBJECTS #
del [X_train, y_train]
gc.collect()

################### XGBOOST ############################

########################################
## INSTANTIATE XGBClassifier()
########################################
xgbc = xgb.XGBClassifier(objective='binary:logistic', parameters=parameters, num_boost_round=10)

########################################
## TRAIN MODEL
########################################
model = xgbc.fit(X_train, y_train)

##################################
# FEATURE IMPORTANCE - GRAPHICAL
##################################
xgb.plot_importance(model)

# Predict values for test data with our model from chapter 5 - the results are saved as a Python array
#test_pred = model.predict(data_test).astype(int)

## OR Set custom threshold 
#test_pred = (model.predict_proba(data_test)[:,1] >= 0.21).astype(int)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictions = [round(value) for value in test_pred]
accuracy = accuracy_score(test_pred, predictions)

pip install scikit-plot

from sklearn.metrics import f1_score, classification_report
from scikitplot.metrics import plot_confusion_matrix
from scikitplot.classifiers import plot_feature_importances

#Evaluation.
print('Result on Validation set for XGBoost')
print('=================================================')
print('F1 Score: {}'.format(f1_score(test_pred, data_test)))
print('=================================================')
print(classification_report(test_pred, data_test))

print('=================================================')
print('Confusion Matrix')
cm = metrics.confusion_matrix(test_pred, data_test)
print(cm)

plot_confusion_matrix(test_pred, data_test)




################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################


"""# 5. Apply predictive model (predict)
The model that we have created is stored in the **model** object.
At this step we predict the values for the test data and we store them in a new column in the same DataFrame.
"""
#for Random Forest
# Predict values for test data with our model from chapter 5 - the results are saved as a Python array
test_pred = model.predict(data_test).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array

## OR set a custom threshold (in this problem, 0.21 yields the best prediction)
test_pred = (model.predict_proba(data_test)[:,1] >= 0.21).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array

#Save the prediction (saved in a numpy array) on a new column in the data_test DF
data_test['prediction'] = test_pred
data_test.head(10)

#Reset the index
final = data_test.reset_index()
#Keep only the required columns to create our submission file (for chapter 6)
final = final[['product_id', 'user_id', 'prediction']]

gc.collect()
final.head()

###For XGBoost

# Predict values for test data with our model from chapter 5 - the results are saved as a Python array
test_pred = model.predict(data_test).astype(int)

## OR Set custom threshold 
test_pred = (model.predict_proba(data_test)[:,1] >= 0.21).astype(int)

test_pred[0:10] #display the first 10 predictions of the numpy array

#Save the prediction (saved in a numpy array) on a new column in the data_test DF
data_test['prediction'] = test_pred
data_test.head(10)

# Reset the index
final = data_test.reset_index()
# Keep only the required columns to create our submission file (for chapter 6)
final = final[['product_id', 'user_id', 'prediction']]

gc.collect()
final.head()