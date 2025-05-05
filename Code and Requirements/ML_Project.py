# Cell
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from scipy.sparse import hstack

# Cell
#Load the Sentiment140 dataset 
data = pd.read_csv('sentiment140.csv', encoding='latin1', header=None)

#Label columns
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']


#Limit the dataset to 20,000 samples in this case to run quicker
data_sampled = data.sample(n=20000, random_state=21).copy()
#standardize sentiment labels to binary
data_sampled['sentiment'] = data_sampled['sentiment'].map({0: 0, 4: 1})

# Cell
# Visualize class distribution
sns.countplot(x='sentiment', data=data_sampled)
plt.title('Class Distribution (Sentiment)')
plt.show()

class_counts = data_sampled['sentiment'].value_counts()

# Print counts
print("Class counts:")
print(class_counts)

# Calculate and print the difference
difference = abs(class_counts[0] - class_counts[1])
print(f"\nDifference: {difference}")

# Cell
#Print head of dataset to see how it is formatted
print("These are the first 5 rows in the dataset. The dataset has the labled sentiment value as 0: negative and 4: positive \n along with other information about the message sent like it's ID, the user who sent it and the date sent.")
print(data_sampled.head())
print("\n ---------------------------------------------------------- \n")

# Cell
#remove common stopwords, apply stemming to word endings, remove @s and links, remove non-alphabetic characters, make all words lowercase
def clean_text(text):
    
    # List of common words to remove 
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by',
        'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of',
        'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there',
        'these', 'they', 'this', 'to', 'was', 'will', 'with', 'you', 'u',
        'your', 'i', 'me', 'my', 'we', 'our', 'he', 'she', 'him', 'her', 
        'now', 'im', 'go', 'going', 'today', 'day', 'know', 'got', 
        'dont', 'one', 'time', 'twitter', 'back', 'look', 'think', 'well'
    }

    # Remove unneccesary word suffixess
    def stem(word):
        if word.endswith('ing') and len(word) > 4:
            return word[:-3]
        elif word.endswith('ed') and len(word) > 3:
            return word[:-2]
        elif word.endswith('es') and len(word) > 3:
            return word[:-2]
        elif word.endswith('ly') and len(word) > 3:
            return word[:-2]
        return word

    # Remove @mentions and URLs
    words = text.split()
    words = [word for word in words if not word.startswith('@') and not word.startswith('http') and not word.startswith('www')]
    text = ' '.join(words)

    # Remove non-alphabetic characters
    cleaned = ''.join(char if char.isalpha() or char.isspace() else '' for char in text)

    # Lowercase, remove stopwords, and apply stemming
    cleaned_words = [stem(word) for word in cleaned.lower().split() if word not in stopwords]

    return ' '.join(cleaned_words)


# Cell
data_sampled['clean_text'] = data_sampled['text'].apply(clean_text)
tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1,2))
X = tfidf.fit_transform(data_sampled['clean_text'])
y = data_sampled['sentiment']

print(data_sampled[['text', 'clean_text']].head())

# Cell
# Split data by sentiment
positive_text = ' '.join(data_sampled[data_sampled['sentiment'] == 1]['clean_text'].astype(str).tolist())
negative_text = ' '.join(data_sampled[data_sampled['sentiment'] == 0]['clean_text'].astype(str).tolist())

# Generate the word clouds
wordcloud_pos = WordCloud(width=1200, height=600).generate(positive_text)
wordcloud_neg = WordCloud(width=1200, height=600).generate(negative_text)

# Set size of wordclouds and display them side by side
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_pos)
plt.axis('off')
plt.title('Positive Sentiment Word Cloud')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_neg)
plt.axis('off')
plt.title('Negative Sentiment Word Cloud')
plt.show()


# Cell
#split training and test data, 80% training 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Cell
# Naive Bayes Setup
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)
pred_nb = naive_bayes_model.predict(X_test)
#print accuracy and use classification report for detailed results
print("Naive Bayes Accuracy:", accuracy_score(y_test, pred_nb))
print("Classification Report (Naive Bayes):\n", classification_report(y_test, pred_nb))


# Cell
# Get feature names from TF-IDF vectorizer
feature_names = tfidf.get_feature_names_out()

# Get the log probabilities from Naive Bayes for each feature/word
log_probs = naive_bayes_model.feature_log_prob_
log_prob_diff = log_probs[1] - log_probs[0]

# Sort by difference
sorted_indices = np.argsort(log_prob_diff)

# Top 20 negative sentiment words
print("Top 20 negative sentiment words (Naive Bayes):")
for i in sorted_indices[:20]:
    print(f"{feature_names[i]}: {log_prob_diff[i]:.2f}")

# Top 20 positive sentiment words
print("\nTop 20 positive sentiment words (Naive Bayes):")
for i in sorted_indices[-20:][::-1]:
    print(f"{feature_names[i]}: {log_prob_diff[i]:.2f}")


# Cell
# Logistic Regression Setup
log_regression_model = LogisticRegression(max_iter=1000)
log_regression_model.fit(X_train, y_train)
pred_lr = log_regression_model.predict(X_test)
#print log regression accuracy and use classification report for detailed results
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, pred_lr))

# Cell
# Get feature names from TF-IDF vectorizer
feature_names = tfidf.get_feature_names_out()

# Get the coefficients from the logistic regression model
coefficients = log_regression_model.coef_[0]

# Sort features by their coefficient values
sorted_indices = np.argsort(coefficients)

# Top 20 features most associated with negative sentiment
print("Top 20 negative sentiment words: (Logistic Regression)")
for i in sorted_indices[:20]:
    print(f"{feature_names[i]}: {coefficients[i]:.2f}")

# Top 20 features most associated with positive sentiment
print("\nTop 20 positive sentiment words: (Logistic Regression)")
for i in sorted_indices[-20:][::-1]:
    print(f"{feature_names[i]}: {coefficients[i]:.2f}")

# Cell
# Confusion matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Confusion matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Reds', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Naive Bayes')
plt.show()


