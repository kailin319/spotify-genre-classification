# Spotify Genre Classification
## Tools & Languages
- Python (Pandas, scikit-learn, matplotlib)
- Jupyter Notebook

## Project Summary
<img width="392" height="107" alt="image" src="https://github.com/user-attachments/assets/ece918a9-bd25-4c68-ab17-6c47e752a3f4" />

This project develops a robust machine learning pipeline to classify Spotify tracks into genres based on audio features. Using a comprehensive dataset of 30,000 songs, we explore data preprocessing, feature engineering, and model training to achieve accurate music genre classification. This study contributes to music information retrieval by advancing feature engineering techniques, benchmarking model performance, and developing a scalable pipeline for large-scale music classification tasks.

## Problem Statement
The problem we are addressing is the classification of music tracks based on their audio features. Specifically, we aim to predict categories such as genre based on various audio attributes like tempo, key, loudness, and danceability.
Spotify, with its vast library, provides a great opportunity to apply such classification models.

## Significance of the Problem
- **Music Recommendation:** Accurate classification helps improve music recommendation systems, making them more personalized and relevant.
- **Music Discovery:** By categorizing tracks efficiently, we can introduce users to music they may not have encountered otherwise, enhancing their listening experience.
- **Industry Application:** For the music industry, understanding trends and categorizing music effectively can help in market analysis, artist promotion, and playlist curation.

## Data Analysis
### Data Preprocessing and EDA
- **Data Cleaning:** Removed irrelevant features and extracted `year` from `album release date`.
- **Feature Engineering:** Created new features from existing data to improve model performance.
- **Exploratory Analysis:** Generated visualizations to understand distribution of genres, popularity, and audio features.

The preprocessing phase laid the foundation for our analysis, ensuring data quality and relevance. Our exploratory data analysis revealed insights into the dataset's characteristics, guiding our subsequent modeling approaches.

### Key Dataset Insights
**Genre Distribution:** Visualization revealed the relative prevalence of different music genres in the dataset.
<img width="983" height="421" alt="image" src="https://github.com/user-attachments/assets/81ec563b-43cc-40f1-a2dc-44968b7dcc9f" />


**Popularity Trends:** Analysis of song popularity over time showed interesting patterns in listener preferences.
<img width="796" height="338" alt="image" src="https://github.com/user-attachments/assets/ed04388d-4f25-41c3-8adf-c29c2bfc525a" />


**Feature Correlations:** Correlation matrix highlighted relationships between various audio features.
<img width="804" height="625" alt="image" src="https://github.com/user-attachments/assets/d21a0aee-1d91-407b-a9e3-e2f63805eccc" />


## Machine Learning 
### Models
- **Random Forest:** Ensemble learning method known for handling high-dimensional data and robustness to overfitting.
- **XGBoost:** Optimized gradient boosting algorithm for enhanced classification accuracy.
- **Neural Network:** Deep learning approach to capture intricate patterns in the data.
We implemented three distinct machine learning models to classify Spotify tracks into genres. Each model offers unique strengths in handling complex audio feature data.

### Model Performance Comparison
- **Random Forest:** Accuracy achieved with optimized hyperparameters, 0.58.
- **XGBoost:** Highest accuracy, outperforming other models, 0.591.
- **Neural Network:** Test accuracy, showing potential for improvement, 0.546.
<img width="797" height="475" alt="image" src="https://github.com/user-attachments/assets/18623dab-f4de-4e55-a92f-0473129dbdc6" />

> Class: edm = 0, latin = 1, pop = 2, r&b = 3, rap = 4, rock = 5


We implemented three distinct machine learning models to classify Spotify tracks into genres. Each model offers unique strengths in handling complex audio feature data.
**XGBoost** demonstrated superior performance in genre classification, highlighting its effectiveness in handling complex, multi-class tasks. The Random Forest model also showed strong results, while the Neural Network approach indicates potential for further optimization.


## Result: Feature Importance
1. **Top Features:** `Year`, `Speechiness`, `Tempo`, `Danceability`
2. **Moderate Impact:** `Energy`, `Loudness`, `Valence`
3. **Lower Influence:** `Mode`, `Key`, `Duration`

The analysis revealed that certain features played a crucial role in genre classification. Understanding feature importance helps refine models and provides insights into the key elements that define music genres.


## Key Findings and Implications
- **XGBoost Superiority:** Demonstrated best performance in genre classification.
- **Temporal Relevance:** Release year proved crucial in genre prediction.
- **Feature Engineering Impact:** Careful feature selection significantly improved model accuracy.
- **Scalable Pipeline:** Developed approach applicable to large-scale music datasets.

The findings have significant implications for music streaming platforms, enabling more accurate personalized recommendations, playlist curation, and music discovery features.

## Future Directions
1. **Advanced Deep Learning:** Explore convolutional and recurrent neural networks for improved accuracy.
2. **Additional Features:** Incorporate lyrics and contextual information for richer analysis.
3. **Cross-Platform Integration:** Extend the model to classify music across different streaming platforms.
4. **Real-Time Classification:** Develop systems for instant genre classification of new releases.

The future of music classification holds exciting possibilities. By continuing to refine our approaches and incorporate new data sources, we can further enhance the accuracy and applicability of genre classification models.
