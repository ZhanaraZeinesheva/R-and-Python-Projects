#!/usr/bin/env python
# coding: utf-8

# # British Airways Case Study with Python
# 
# ### Zhanara Zeinesheva
# ### 09/06/2024

# # Task 2
# 
# ---
# 
# ## Predictive modeling of customer bookings
# 
# This Jupyter notebook includes some code to get started with this predictive modeling task. i will use various packages for data manipulation, feature engineering and machine learning.
# 
# ### Exploratory data analysis
# 
# First, I explore the data in order to better understand what we have and the statistical properties of the dataset.

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")
df.head()


# The `.head()` method allows us to view the first 5 rows in the dataset, this is useful for visual inspection of our columns

# In[3]:


df.info()


# The `.info()` method gives us a data description, telling us the names of the columns, their data types and how many null values we have. Fortunately, I have no null values. It looks like some of these columns should be converted into different data types, e.g. flight_day.
# 
# To provide more context, below is a more detailed data description, explaining exactly what each column means:
# 
# - `num_passengers` = number of passengers travelling
# - `sales_channel` = sales channel booking was made on
# - `trip_type` = trip Type (Round Trip, One Way, Circle Trip)
# - `purchase_lead` = number of days between travel date and booking date
# - `length_of_stay` = number of days spent at destination
# - `flight_hour` = hour of flight departure
# - `flight_day` = day of week of flight departure
# - `route` = origin -> destination flight route
# - `booking_origin` = country from where booking was made
# - `wants_extra_baggage` = if the customer wanted extra baggage in the booking
# - `wants_preferred_seat` = if the customer wanted a preferred seat in the booking
# - `wants_in_flight_meals` = if the customer wanted in-flight meals in the booking
# - `flight_duration` = total duration of flight (in hours)
# - `booking_complete` = flag indicating if the customer completed the booking
# 
# Before I compute any statistics on the data, lets do any necessary data conversion

# In[4]:


df["flight_day"].unique()


# In[5]:


mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}

df["flight_day"] = df["flight_day"].map(mapping)


# In[6]:


df["flight_day"].unique()


# In[7]:


df.describe()


# The `.describe()` method gives us a summary of descriptive statistics over the entire dataset (only works for numeric columns). This gives us a quick overview of a few things such as the mean, min, max and overall distribution of each column.
# 
# From this point, I continue exploring the dataset with some visualisations and other metrics that I think may be useful. Then, I prepare the dataset for predictive modelling. Finally, I train my machine learning model, evaluate it with performance metrics and output visualisations for the contributing variables.

# #### Distribution Plots

# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['purchase_lead'], kde=True)
plt.title('Distribution of Purchase Lead Time')
plt.show()

sns.boxplot(x=df['length_of_stay'])
plt.title('Boxplot of Length of Stay')
plt.show()


# Distribution of Purchase Lead Time: 
# * The histogram shows that most bookings are made very close to the flight date, with a significant number of bookings occurring within 0 to 50 days before the flight. 
# * There is a steep drop-off after 50 days, meaning that few people book flights more than 100 days in advance. 
# * The distribution is right-skewed, indicating that while most bookings are made closer to the departure date, a smaller number of outliers are booking far in advance (e.g., more than 400 or even 800 days).
# 
# Boxplot of Length of Stay: 
# * The boxplot highlights a wide range in the length of stay with a significant number of outliers.
# * The majority of the stays are clustered below 50 days, but there are many outliers with very long stays, some extending up to 800 days.
# * The presence of numerous outliers suggests there may be unusual or extreme cases where customers are staying much longer than typical passengers, which could warrant further investigation.

# In[9]:


sns.countplot(x='wants_extra_baggage', data=df)
plt.title('Count of Extra Baggage Requests')
plt.show()


# Most customers want an extra baggage.

# In[10]:


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# * Wants Preferred Seat and Wants In-flight Meals: There is a moderate correlation of 0.32 between customers who prefer a seat and those who want in-flight meals, suggesting that customers who have seat preferences might also prefer additional in-flight services.
# * Wants Extra Baggage and Wants Preferred Seat: A moderate correlation of 0.21 between passengers wanting extra baggage and wanting preferred seating implies that customers who opt for extra baggage are more likely to also choose seat preferences.
# * Length of Stay and Wants Extra Baggage: There is a slight positive correlation of 0.18, indicating that passengers who stay longer may be more inclined to carry extra baggage.

# In[11]:


sns.scatterplot(x='purchase_lead', y='flight_hour', data=df)
plt.title('Purchase Lead vs Flight Hour')
plt.show()


# Most bookings are made within a short purchase lead time (around 0-100 days). Beyond that, the number of bookings drops significantly.

# In[12]:


sns.violinplot(x='wants_extra_baggage', y='length_of_stay', data=df)
plt.title('Length of Stay vs Extra Baggage Request')
plt.show()


# Both groups, with and without extra baggage requests, have similar distributions, with most passengers staying for less than 50 days.
# The presence of outliers with very long stays suggests some passengers have significantly longer trips, but this does not differ based on extra baggage requests.

# In[13]:


df['flight_day'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(7, 7))
plt.title('Flight Day Distribution')
plt.show()


# Flight activity is relatively balanced throughout the week, with a slight increase on Monday, Tuesday and Wednesday and a slight dip on Saturday. 

# In[14]:


sns.pairplot(df[['num_passengers', 'purchase_lead', 'length_of_stay', 'flight_hour']])
plt.title('Pairwise Relationships')
plt.show()


# There is a notable negative relationship between purchase lead and length of stay, with shorter stays being booked further in advance. As the number of days in advance the flight is booked increases, the length of stay tends to decrease. The plot shows that customers with shorter stays tend to book further in advance, while those with longer stays tend to book closer to their departure date.
# 
# Other relationships, such as between the number of passengers, flight hour, and length of stay, do not exhibit strong correlations. Most passengers are solo or travel in pairs, and flights are fairly evenly distributed throughout the day.

# In[15]:


#creating of new features
df['passengers_per_lead'] = df['num_passengers'] / (df['purchase_lead'] + 1)
df['last_minute_booking'] = df['purchase_lead'].apply(lambda x: 1 if x < 7 else 0)
df['total_requests'] = df['wants_extra_baggage'] + df['wants_preferred_seat'] + df['wants_in_flight_meals']


# In[16]:


# Define X (features) and y (target)
X = df.drop(columns=['booking_complete', 'flight_day', 'wants_preferred_seat', 'wants_in_flight_meals', 'flight_hour'])
y = df['booking_complete']  # target column for predicting bookings


# #### Spliting the Data Into Training and Testing Sets

# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[19]:


df_encoded = pd.get_dummies(df, drop_first=True) #Encode Categorical Features


# In[20]:


# Define target and features
X = df_encoded.drop('booking_complete', axis=1)
y = df_encoded['booking_complete']

# Split into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# #### Train the RandomForest Model

# In[21]:


# Initialize the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)


# #### Make Predictions

# In[23]:


# Make predictions
y_pred = model.predict(X_test)


# #### Evaluate the Model

# In[24]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# Print classification report
print(classification_report(y_test, y_pred))


# The model has an overall accuracy of 85%, which means it correctly predicts whether a customer completes a booking most of the time.
# 
# For customers who don't complete bookings (class 0), the model does very well with high precision (87%) and recall (98%), meaning it’s almost always correct.
# For customers who do complete bookings (class 1), the model struggles. It only correctly identifies 12% of them and is accurate 51% of the time when predicting a completed booking.

# #### Check the Importance

# In[25]:


import matplotlib.pyplot as plt
import numpy as np

# Get feature importances
importances = model.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[-10:]  # top 10 features

# Plot
plt.figure(figsize=(10, 6))
plt.title('Top 10 Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# Passengers per lead and Purchase lead are the most influential factors, meaning they have the highest impact on predicting a booking.
# 
# Other significant features include Flight hour, Length of stay, and Flight day, all of which play a key role in determining booking behavior.
# 
# Location-based factors like Booking origin from Malaysia and Australia also contribute but have less influence compared to time-based and passenger-related features.

# #### Cross-Validation

# In[26]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation (with 5 folds)
cv_scores = cross_val_score(model, X, y, cv=5)

# Print the average accuracy
print(f'Average Cross-Validation Accuracy: {np.mean(cv_scores):.4f}')


# #### Adjusting for Class Imbalance

# In[27]:


model = RandomForestClassifier(class_weight='balanced', random_state=42)


# #### Retrain the Model

# In[29]:


X = df_encoded.drop('booking_complete', axis=1)  # Features
y = df_encoded['booking_complete']  # Target

# Make sure to use the same split for both X and y
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Now I should have consistent lengths between X_train_encoded and y_train


# In[30]:


print(X_train_encoded.shape)  # Check dimensions of X_train_encoded
print(y_train.shape)  # Check dimensions of y_train


# In[31]:


# Train the model with class weights
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_encoded, y_train)


# #### Evaluate the Model

# In[32]:


# Make predictions
y_pred = model.predict(X_test_encoded)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# Print classification report
print(classification_report(y_test, y_pred))


# The model is more effective at predicting when a booking is not made, but it needs improvement in identifying when a booking does happen.

# #### Check Feature Importance

# In[33]:


importances = model.feature_importances_
feature_names = X_train_encoded.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot top 10 feature importances
feature_importance_df.head(10).plot(kind='bar', x='Feature', y='Importance', title='Top 10 Feature Importances')
plt.show()


# Passengers per lead and Purchase lead are the two most important factors, having the greatest influence on the model's predictions.
# 
# Other key factors include Length of stay and Flight hour, which also play significant roles in determining booking behavior.
# 
# Features like Flight day, Booking origin from Australia and Malaysia, and Flight duration have moderate importance.
# 
# The number of passengers and the total requests feature, while included in the top 10, contribute relatively less compared to other features.

# #### Cross-Valudation

# In[34]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation (with 5 folds)
cv_scores = cross_val_score(model, X, y, cv=5)

# Print the average accuracy
print(f'Average Cross-Validation Accuracy: {np.mean(cv_scores):.4f}')


# ## Recommendations

# Based on the Top 10 Feature Importances plot, here are a few recommendations to increase completed bookings:
# 
# * Optimize Passengers per Lead: Since this feature is highly important, consider offering group discounts or incentives for multiple passengers booking together to encourage more bookings.
# 
# * Improve Purchase Lead Times: Offer promotions for early bookings, such as discounts or special offers, to encourage customers to book well in advance. Also, for last-minute bookings, providing seamless and fast booking experiences could reduce drop-offs.
# 
# * Adjust for Flight Hour Preferences: Since flight hour plays a key role, it might help to ensure flights are scheduled during the most preferred hours, potentially aligning with customer demand for specific time windows.
# 
# * Maximize Length of Stay Discounts: Provide flexible booking options or discounts for different lengths of stay to encourage bookings for various trip durations. Target specific marketing toward customers with varying stay lengths.
# 
# * Tailor Regional Offers: For customers from Australia and Malaysia, who are also influential based on the booking origin features, offer targeted promotions or loyalty programs to increase the likelihood of bookings.
