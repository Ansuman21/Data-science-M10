# CriticEye: Automating Sentiment Analysis for Movie Reviews

## Project Overview

In this project, we aim to perform sentiment analysis on movie reviews using natural language processing (NLP) and machine learning techniques. We specifically utilize the Naïve Bayes classifier, a widely used algorithm for text classification tasks. The goal is to predict the sentiment of movie reviews (positive or negative) based on the content of the reviews.

## Business Case

In today’s digital age, businesses and media organizations are flooded with large volumes of user-generated content such as reviews, comments, and feedback. Analyzing these reviews can provide valuable insights into customer satisfaction, brand perception, and market trends. By automating the sentiment analysis of movie reviews, we can:

- **Gauge public opinion**: Quickly determine whether reviews are positive or negative, offering immediate feedback on movie reception.
- **Improve decision-making**: Help movie studios and production houses make data-driven decisions about movie releases, promotions, or adjustments to films.
- **Enhance customer engagement**: Enable platforms like IMDB, Rotten Tomatoes, or Netflix to offer personalized recommendations based on sentiment trends in reviews.

## Key Steps Taken

### 1. **Data Collection & Preprocessing**
   - **Data Collection**: A dataset containing positive and negative movie reviews was collected. 
   - **Text Preprocessing**: 
     - Removed stopwords, punctuation, and special characters.
     - Tokenized the text data and performed stemming or lemmatization to reduce words to their root forms.
     - The text data was then transformed into a format suitable for model training.

### 2. **Feature Extraction**
   - **Term-Document Matrix Construction**: Used `CountVectorizer` to create a term-document matrix that represents the frequency of each word in the reviews. This matrix serves as the input feature set for the machine learning model.

### 3. **Data Splitting**
   - The data was split into **training** (75%) and **testing** (25%) subsets using `train_test_split` to train the model on one portion of the data and evaluate it on another.

### 4. **Model Selection & Training**
   - **Naïve Bayes Classifier**: A **Multinomial Naïve Bayes** classifier was chosen for text classification because it performs well with word frequency data.
   - The model was trained on the training subset of the data, allowing it to learn the relationships between words and sentiment.

### 5. **Model Evaluation**
   - The model's performance was evaluated using **accuracy**, as well as more detailed metrics like **precision**, **recall**, and **F1-score** for both positive and negative classes.
   - The Naïve Bayes classifier achieved an accuracy of **81.2%** on the testing data.

### 6. **Feature Importance**
   - The top 30 most informative features were identified, revealing the words that had the most significant influence on the sentiment classification.
   - Words like **"film"**, **"movie"**, **"like"**, **"story"**, and **"life"** were identified as key drivers of sentiment in the reviews.

### 7. **Deployment & Prediction**
   - The model was deployed to classify an **unseen review**. The review was correctly classified as **negative** based on the model’s analysis, showcasing its ability to generalize to new, unseen data.

## Business Recommendations

1. **Sentiment-Driven Decisions**: Movie studios can use the sentiment analysis model to understand audience reactions to their films and make real-time decisions regarding marketing, promotional efforts, or even editing and reshooting scenes based on feedback.
   
2. **Improving Recommendation Systems**: Platforms like Netflix can use sentiment analysis to improve their recommendation algorithms. By analyzing the sentiment of reviews, the platform can suggest movies that align with the user's tastes and preferences, improving customer satisfaction.

3. **Real-Time Feedback Loop**: By automating sentiment analysis, businesses can receive real-time insights into how their content or products are being perceived, allowing them to act swiftly to address concerns or capitalize on positive feedback.

4. **Exploring More Complex Models**: While Naïve Bayes provides a solid baseline, exploring advanced models like **Logistic Regression**, **SVM**, or even **Deep Learning** models can help improve accuracy and handle more nuanced sentiment.

5. **Contextual Sentiment Analysis**: To further improve the model, incorporate word embeddings (e.g., Word2Vec, GloVe) or transformer models (e.g., BERT) to capture contextual sentiment more accurately.

## Conclusion

The project demonstrated the application of the Naïve Bayes classifier in sentiment analysis of movie reviews. The model performed reasonably well, achieving an accuracy of **81.2%**, and was able to classify unseen reviews accurately. This solution can be a valuable tool for businesses looking to understand customer sentiment and make data-driven decisions based on public feedback.

## Author

**Ansuman Patnaik**  
MS in Data Science & Analytics, Yeshiva University  
Email: ansu1p89k@gmail.com
