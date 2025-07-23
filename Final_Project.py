#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


# # Data Preprocessing

# In[2]:


file = "data/spotify_songs.csv"
df = pd.read_csv(file)
df.head(5)


# In[3]:


df.info()
df.describe()


# Convert the release date to year

# In[4]:


# Convert 'track_album_release_date' to datetime, using errors='coerce' to handle invalid dates
df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')

# Handle cases where the date is just a year (e.g., '2012')
df['track_album_release_date'] = df['track_album_release_date'].fillna(pd.to_datetime(df['track_album_release_date'], format='%Y'))

# Extract the year and create a new column 'year'
df['year'] = df['track_album_release_date'].dt.year
df['year'] = df['year'].astype('Int64')
df.head(5)


# In[5]:


del df['track_id'], df['track_name'], df['track_album_id'], df['track_album_name'], df['track_album_release_date'], df['playlist_name'], df['playlist_id']
del df['track_artist']
df.info()


# In[56]:


year_count = df["year"].value_counts()
#print(year_count)
plt.figure(figsize=(20,8))
sns.countplot(x='year', data=df, color="#66b3ff")
plt.title('Number of songs by year')
plt.show()


# In[45]:


plt.figure(figsize=(20, 8))
sns.lineplot(x='year', y='duration_ms', data=df)
plt.title('Song duration over time')
plt.show()


# In[49]:


genre_count = df['playlist_genre'].value_counts()
print(genre_count)

plt.figure(figsize=(20,8))
sns.countplot(x='playlist_genre', data=df, palette='Set2')
plt.title('Number of songs by genre')
plt.show()


# In[52]:


plt.figure(figsize=(20,8))
sns.countplot(x='playlist_subgenre', data=df, palette='Set3')
plt.xticks(rotation=90)
plt.title('Number of songs by subgenre')
plt.show()


# In[55]:


plt.figure(figsize=(20,8))
plt.hist(df['track_popularity'], bins='auto', color='#66b3ff', edgecolor='black')
plt.xlabel('Popularity')
plt.ylabel('Frecuency')
plt.title('Distribution of popularity of songs')
plt.show()


# Get Hot-Label for genre

# In[9]:


df.info()


# In[24]:


le = LabelEncoder()
df_for_model = df.drop(df.select_dtypes(include=['object']) , axis=1)
df_for_model['playlist_genre'] = le.fit_transform(df['playlist_genre'])
df_for_model.head()


# In[25]:


print(genre_count, df_for_model['playlist_genre'].value_counts())


# The labels with the original genre.
# - edm = 0
# - latin = 1
# - pop = 2
# - r&b = 3
# - rap = 4
# - rock = 5

# In[65]:


# Correlation Matrix
corr_matrix = df_for_model.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Matrix')
plt.show()


# In[62]:


fig, axs = plt.subplots(3, 2, figsize=(20, 15))

features = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence']

for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    ax = axs[row][col]

    # Plot the histogram
    n, bins, patches = ax.hist(df_for_model[feature], bins=20, edgecolor='black', alpha=0.5)
    ax.set_title(feature)

    # Calculate and plot the mean
    mean = np.mean(df_for_model[feature])
    std = np.std(df_for_model[feature])

    ax.axvline(mean, color='red', linestyle='dashed', linewidth=1)
    ax.text(mean, max(n) * 0.9, f'Mean: {mean:.2f}', color='red', ha='center')

    for patch in patches:
        patch.set_edgecolor('black')

plt.tight_layout()
plt.show()


# # Data Split

# By some features

# In[110]:


train_df = df[df['year'] < 2019]
test_df = df[df['year'] >= 2019]
print(f"Training: {len(train_df)}, Testing:{len(test_df)}")


# In[ ]:


# Featuring
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
X_train = train_df[features]
y_train = train_df['playlist_genre']
X_test = test_df[features]
y_test = test_df['playlist_genre']


# In[127]:


emo_features = ['danceability', 'energy', 'loudness', 'valence', 'tempo']
X_train = train_df[emo_features]
y_train = train_df['playlist_genre']
X_test = test_df[emo_features]
y_test = test_df['playlist_genre']


# Use All the Features

# In[26]:


X = df_for_model.drop('playlist_genre', axis=1)
# create dataset Y with only playlist_genre
y = df_for_model['playlist_genre']
# split data into train and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    train_size = 0.75,
    test_size = 0.25,
    random_state=10
)


# In[27]:


print(f"X(train, test): {len(X_train),len(X_test)}, y(train, test): {len(y_train), len(y_test)}")


# # Random Forest Tree

# Use emotional features

# In[128]:


# emo_features
# Training Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Testing Model
y_pred = rf_model.predict(X_test)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Report
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# Use part of features

# In[ ]:


# Training Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Testing Model
y_pred = rf_model.predict(X_test)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Report
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# In[ ]:


# Define parameter
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model
print(f"Best Parameters: {grid_search.best_params_}")
best_rf_model = grid_search.best_estimator_

# Test Best Model
y_pred_optimized = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred_optimized))


# In[171]:


# Train model and observe the feature importance
model = grid_search.best_estimator_
feature_importances = model.feature_importances_

# Visualization
features = X_train.columns
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(8, 4))
plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
plt.title("Feature Importances")
plt.show()


# Use All the features

# In[29]:


forestModel = RandomForestClassifier()
forestModel.fit(X_train,y_train)
y_pred = forestModel.predict(X_test)


# In[30]:


print(f"Accuracy:{accuracy_score(y_test,y_pred):.2f}")


# In[31]:


scores = cross_val_score(forestModel, X_train, y_train, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean Score:", np.mean(scores))
print("Standard Deviation:", np.std(scores))


# In[32]:


# Training Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Testing Model
y_pred = rf_model.predict(X_test)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Report
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# In[33]:


from sklearn.model_selection import GridSearchCV

# Define parameter
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10, 15]
}

# GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model
print(f"Best Parameters: {grid_search.best_params_}")
best_rf_model = grid_search.best_estimator_

# Test Best Model
y_pred_optimized = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred_optimized))


# In[34]:


# Define parameter
param_grid = {
    'n_estimators': [100, 200, 300], # Larger estimators
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10, 15]
}

grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model
print(f"Best Parameters: {grid_search.best_params_}")
best_rf_model = grid_search.best_estimator_

# Test Best Model
y_pred_optimized = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred_optimized))


# In[ ]:


# Train model and observe the feature importance
feature_importances = best_rf_model.feature_importances_

# Visualization
features = X_train.columns
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(8, 4))
plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
plt.title("Feature Importances")
plt.show()


# # Neuronal Network

# In[177]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# In[178]:


# Define the features and target
X = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
y =df['playlist_genre']

# Convert the target to categorical
# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')


# In[179]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')


# # XGBoost

# In[28]:


from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Instantiate the model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)


# # Comparison

# In[53]:


# Model
models = ['Random Forest', 'Neural Network', 'XGBoost']

# Precision
precision_rf = [0.69, 0.51, 0.41, 0.51, 0.60, 0.75]  # Random Forest precision
precision_xgb = [0.71, 0.51, 0.43, 0.50, 0.61, 0.77]  # XGBoost precision

# Labels
classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']

# Neural Network
accuracy_nn = 0.546  # 假設 Neural Network 準確率是 0.546

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.25
index = np.arange(len(classes))

# Random Forest precision
bars1 = ax.bar(index - bar_width, precision_rf, bar_width, label='Random Forest', color='#ff9999')

# XGBoost precision
bars2 = ax.bar(index, precision_xgb, bar_width, label='XGBoost', color='#c2c2f0')

# Neural Network accuracy
bars3 = ax.bar(index + bar_width, [accuracy_nn] * len(classes), bar_width, label='Neural Network Accuracy', color='#66b3ff')


ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Precision Comparison by Class for Different Models (NN Accuracy Shown)')
ax.set_xticks(index)
ax.set_xticklabels(classes)


ax.legend()


plt.tight_layout()
plt.show()


# In[54]:


models = ['Random Forest', 'XGBoost', 'Neural Network']

# acc
#accuracies = [0.59, 0.546, 0.591]
accuracies = [0.59, 0.591, 0.546]

# F1-score (weighted avg)
#f1_scores = [0.58, 0, 0.59]  
f1_scores = [0.58, 0.59, 0]  


# plot
fig, ax = plt.subplots(figsize=(10, 6))


bar_width = 0.35
index = np.arange(len(models))
bars1 = ax.bar(index, accuracies, bar_width, label='Accuracy', color='#66b3ff')
bars2 = ax.bar(index + bar_width, f1_scores, bar_width, label='F1-score', color='#c2f0c2')

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Performance')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models)


ax.legend()


plt.tight_layout()
plt.show()

