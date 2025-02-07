# 🎵 **Song Genre Classification and Sentiment Analysis** 🎶

## 📚 Project Overview

This project focused on classifying song genres based on lyrics and analyzing the sentiment of those lyrics. The dataset consists of song lyrics labeled with genres, and the goal was to predict the genre of a song given its lyrics. Additionally, the sentiment analysis explores the emotional undertones of the lyrics and identifies any notable sentiment trends across different genres and over time.

This was a group project and was done for the **Text Mining** course.

<br>

---

## 📂 Dataset

The dataset used for this project contains song lyrics from various genres. The dataset contains multiple genres, but the distribution is highly **imbalanced**, with some genres (e.g., **Pop**) appearing much more frequently than others (e.g., **Country**, **Jazz**).

### 📝 **Features:**

1. **`id`**: Unique identifier for each song.
2. **`title`**: The title of the song.
3. **`artist`**: The artist or group performing the song.
4. **`tag`**: The genre of the song (e.g., Pop, Rap, Rock, Country, etc.).
5. **`lyrics`**: The full text of the song lyrics.
6. **`year`**: The year the song was released.
7. **`views`**: The number of views the song has received on the platform (if applicable).
8. **`feature`**: Additional categorical feature, which might represent something like whether the song is a remix or an acoustic version.

<br>

---

## 🔍 Data Exploration and Preprocessing

### 🧐 Data Inspection

- **Duplicate Values**: Exploring if duplicate values exist, there was none.
- **Missing Values**: The only missing value was in the ‘title’ column (2 missing values). 
- **Data Types**: We identified 5 categorical variables (`title`, `tag`, `artist`, `feature`, `lyrics`) and 2 numerical variables (`year`, `views`).
  
### 📊 Visualization

- Visualized the distribution of genres and found our dataset to be imbalanced, with 'pop' songs making up 41.3% of the dataset, followed by 'rap' (28.6%) and 'rock' (18.8%).
- We also analyzed the number of songs per year and discovered some unusual data entries, such as songs from the year 1.

### 🧹 Text Preprocessing

- **Lyrics Cleaning**: We cleaned the song lyrics by removing unwanted characters, fixing apostrophes, handling special formatting like italics, and eliminating languages other than English. We also added several features to aid classification:
    - `lyrics_len`: Number of characters in the lyrics.
    - `Has_regex_html`: Boolean indicating if the lyrics contain a URL.
    - `new_line_count`: Number of lines in the song lyrics.
- **Genre-Specific Cleaning**: We applied specific cleaning rules for each genre to remove non-song entries, such as essays, interviews, and scientific papers, from the dataset.

<br>

---

## 📊 Genre Classification

### 🔧 Preprocessing for Classification

- After preprocessing the data, we performed feature extraction using the **Bag-of-Words** (BoW) model and **TF-IDF** to represent the song lyrics.
- Several classification models were evaluated, including Logistic Regression, Multinomial Naive Bayes, Decision Trees, and Random Forests.
- **Best Model**: The best-performing model was a combination of **Bag-of-Words** and **Multinomial Naive Bayes**, which achieved an F1-score of **0.5993**.

### ⚙️ Model Optimization

- A **Grid Search** was performed to tune hyperparameters and improve the model’s performance.
- Attempts were made to further improve accuracy using **Doc2Vec** for word embeddings but did not lead to significant improvement. Eventually, **Logistic Regression** with L1 regularization and the inverse regularization strength `C=10` achieved the best result.

<br>

---

## 💬 Sentiment Analysis

We analyzed the emotional undertones of song lyrics using two rule-based approaches:

1. **Vader**: Best suited for informal text like song lyrics, it measures sentiment as positive, neutral, or negative.
2. **TextBlob**: Measures polarity and subjectivity, generally used for more formal texts.

### 📈 Findings

- **General Sentiment**: Most songs had a neutral sentiment, with certain genres (like 'rap') showing a stronger negative sentiment, while 'rb' showed more positive sentiments.
- **Sentiment vs. Views**: Songs with more extreme emotions (either positive or negative) tended to receive more views, especially in the 'rap' genre.
- **Yearly Trends**: Sentiment analysis revealed that certain genres, such as 'rap,' have become more negative over the years, while 'rock' and 'rb' have become more positive.

---

## 🔎 Model Evaluation

- **F1 Score**: The best F1-score obtained for genre classification was **0.608** on the training data.
- **Confusion Matrix**: The model performed well with 'pop' and 'rap' genres but struggled with 'country' and 'rb'.
- **Metrics**: Precision, recall, and F1-scores for various genres were calculated, revealing the model's tendency to favor 'pop' and misclassify genres like 'country' and 'rb'.

---

## 📌 Conclusions

### 🎯 Genre Classification

- Our best classification model was based on **Multinomial Naive Bayes** and **Bag-of-Words**, achieving an F1-score of **0.608**.
- Despite efforts to balance the dataset and optimize the model, the class imbalance still led to bias towards the 'pop' genre.

### 💭 Sentiment Analysis

- Sentiment analysis revealed interesting trends, with **rap** showing the most negative emotions and **rb** being the most positive.
- Extreme sentiment, whether positive or negative, correlated with higher views, indicating that emotional extremes may drive popularity.

### 🚧 Challenges

- Data imbalances and limited computational resources were key challenges throughout the project.
- Despite trying various techniques, the model performance did not exceed an F1-score of **0.61**, and there was a significant bias towards the 'pop' genre.

---

## 🚀 Running the Code

### 1. Clone this repository:
```bash
git clone https://github.com/MGN19/text-mining-lyrics.git
```

## 🗂️ Required Files to Run the Project

### 1. **`shortcuts.py`** 📜
This Python file contains utility functions that are used in the notebooks.

### 2. **`0_Preprocessing.ipynb`** 🧹
This notebook handles the **data preprocessing**.

### 3. **`1_Classification.ipynb`** 🏷️
This notebook performs **genre classification**.

### 4. **`3_SentimentAnalysis.ipynb`** 💬
This notebook performs **sentiment analysis**.

### To Run the Project, Execute the Following Steps:
```bash
# Clone the repository
git clone https://github.com/MGN19/text-mining-lyrics.git
```
# Step 1: Functions needed 
jupyter notebook shortcuts.py

# Run the notebooks in sequence:
# Step 2: Preprocess the data
jupyter notebook 0_Preprocessing.ipynb

# Step 3: Run Classification
jupyter notebook 1_Classification.ipynb

# Step 4: Perform Sentiment Analysis
jupyter notebook 3_SentimentAnalysis.ipynb



