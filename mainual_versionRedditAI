import openpyxl, praw, random,string
import concurrent.futures
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from autocorrect import Speller
from spellchecker import SpellChecker
spell = Speller(lang='en')
spell = SpellChecker()
stopwords = nltk.corpus.stopwords.words('english')
avrigeTitleSize = []
# path
path = r"C:\Users\danie\OneDrive\Desktop\Data.xlsx"
workbook = openpyxl.load_workbook(path)
worksheet = workbook.active
# reddit
reddit = praw.Reddit(
    client_id="nY4WstIZvogjrDxmIWyKPw",
    client_secret="jKZGZhgITUi8rd_xgMtlkscMx62JoA",
    user_agent="test-script",
)
# --- add data
current_row = 2
subreddit_name = "randomscreenshot"
subreddit = reddit.subreddit(subreddit_name)
num_posts = subreddit.subscribers
print(f"Looking through {num_posts} posts....")
# Get existing titles from the Excel file
existing_titles = set(worksheet.cell(row=row, column=1).value for row in range(2, worksheet.max_row + 1))
for submission in reddit.subreddit(subreddit_name).new(limit=None):
    if submission.title not in existing_titles:  # Check if the title already exists
        worksheet = workbook['Sheet1']
        worksheet.cell(row=current_row, column=1, value=submission.title)
        avrigeTitleSize.append(len(submission.title))
        worksheet.cell(row=current_row, column=2, value=submission.score)
        worksheet.cell(row=current_row, column=3, value=submission.num_comments)
        worksheet.cell(row=current_row, column=4, value=len(submission.all_awardings))
        created_utc_timestamp = submission.created_utc
        created_utc_datetime = datetime.utcfromtimestamp(created_utc_timestamp)
        formatted_timestamp = datetime.utcfromtimestamp(created_utc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        worksheet.cell(row=current_row, column=5, value=formatted_timestamp)
        worksheet.cell(row=current_row, column=6, value=submission.url)
        worksheet.cell(row=current_row, column=7, value=submission.link_flair_text)
        workbook.save(path)
        current_row += 1
workbook.close()
# preprocess text
ps = PorterStemmer()
def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [ps.stem(w.lower()) for w in words if w.isalpha() and w.lower() not in stopwords and w not in string.punctuation and not w.isdigit()]
    return " ".join(filtered_words)
# build model
def build_model(max_words, max_text_length):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_text_length))
    model.add(LSTM(100))
    model.add(Dense(200, activation='sigmoid'))  # Linear activation for regression
    model.compile(optimizer='SGD', loss='huber_loss')
    return model
# Function to preprocess user input and get predictions
def get_predictions(user_input, tokenizer, max_text_length, score_model, comments_model, awards_model):
    processed_input = preprocess_text(user_input)
    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([processed_input])
    input_sequence = pad_sequences(input_sequence, maxlen=max_text_length)
    # Get predictions for score, comments, and awards
    score_prediction = score_model.predict(input_sequence)
    comments_prediction = comments_model.predict(input_sequence)
    awards_prediction = awards_model.predict(input_sequence)
    # Calculate average predictions
    avg_score = round(np.mean(score_prediction))
    avg_comments = round(np.mean(comments_prediction))
    avg_awards = round(np.mean(awards_prediction))
    # Calculate standard deviation for each prediction
    std_score = np.std(score_prediction)
    std_comments = np.std(comments_prediction)
    std_awards = np.std(awards_prediction)
    # Calculate confidence interval (assuming normal distribution)
    z_score = 1.00
    confidence_interval_score = (avg_score - z_score * std_score, avg_score + z_score * std_score)
    confidence_interval_comments = (avg_comments - z_score * std_comments, avg_comments + z_score * std_comments)
    confidence_interval_awards = (avg_awards - z_score * std_awards, avg_awards + z_score * std_awards)
    return avg_score, confidence_interval_score, avg_comments, confidence_interval_comments, avg_awards, confidence_interval_awards
# read data
data = pd.read_excel(path)
max_text_length = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Title'])
X_text = pad_sequences(tokenizer.texts_to_sequences(data['Title']), maxlen=max_text_length)
X_text = pad_sequences(X_text, maxlen=max_text_length)
X_numerical = data[['Score', 'Comments', 'Awards']].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Title'])
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.5, random_state=50, shuffle=True)
X_train_text = pad_sequences(tokenizer.texts_to_sequences(X_train['Title']), maxlen=max_text_length)
# Text input model
max_words = len(tokenizer.word_index) + 1
embedding_dim = 128
text_input = Input(shape=(max_text_length,))
text_embedding = Embedding(input_dim=max_words, output_dim=embedding_dim)(text_input)
text_lstm = LSTM(100)(text_embedding)
# Numerical input model
numerical_input = Input(shape=(3,))
numerical_dense = Dense(100)(numerical_input)
# Merge models
merged = concatenate([text_lstm, numerical_dense])
output = Dense(max_words, activation='softmax')(merged) 
model = Model(inputs=[text_input, numerical_input], outputs=output)
model.compile(optimizer=Adam(lr=0.1), loss='huber_loss', metrics=['accuracy'])
model.fit([X_text, X_numerical], y_encoded, epochs=10, batch_size=64, validation_split=0.5)
# User input and prediction
submissionlList = []
NewImageList = []
for submission in reddit.subreddit("meme").top(limit=10):
    submissionlList.append(submission.title)
    NewImageList.append(submission.url)
user_input_text = random.choice(submissionlList)
user_input_text_seq = tokenizer.texts_to_sequences([user_input_text])
user_input_text_padded = pad_sequences(user_input_text_seq, maxlen=max_text_length)
user_input_numerical = np.array([[random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)]])
predicted_word_index = np.argmax(model.predict([user_input_text_padded, user_input_numerical]))
predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_word_index][0]
# Build models for score, comments, and awards
score_model = build_model(max_words, max_text_length)
comments_model = build_model(max_words, max_text_length)
awards_model = build_model(max_words, max_text_length)
X_train_text = pad_sequences(tokenizer.texts_to_sequences(X_train['Title']), maxlen=max_text_length)
# Train and predict for score, comments, and awards
score_model.fit(X_train_text, X_train['Score'].values, batch_size=64, epochs=10, validation_split=0.5)
comments_model.fit(X_train_text, X_train['Comments'].values, batch_size=64, epochs=10, validation_split=0.5)
awards_model.fit(X_train_text, X_train['Awards'].values, batch_size=64, epochs=10, validation_split=0.5)
# Predict for user input
def generate_title():
    TItlelist = []
    for _ in range(random.randint(2, 10)):
        user_input_text_seq = tokenizer.texts_to_sequences([user_input_text])
        user_input_text_padded = pad_sequences(user_input_text_seq, maxlen=max_text_length)
        user_input_numerical = np.array([[random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)]])
        predicted_word_index = np.argmax(model.predict([user_input_text_padded, user_input_numerical]))
        predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_word_index][0]
        user_input_processed = preprocess_text(predicted_word)
        user_input_sequence = tokenizer.texts_to_sequences([user_input_processed])
        user_input_padded = pad_sequences(user_input_sequence, maxlen=max_text_length)
        TItlelist.append(predicted_word)
    return TItlelist
# spell check
def spellCheck(input_string):
    input_string = input_string.lower()
    tokenized_words = word_tokenize(input_string)
    filtered_words = [word for word in tokenized_words if word.isalpha() and word not in stopwords]
    corrected_words = [spell.correction(word) for word in filtered_words]
    deduplicated_words = []
    prev_word = None
    for word in corrected_words:
        if word != prev_word:
            deduplicated_words.append(word)
        prev_word = word
    corrected_string = ' '.join(deduplicated_words)
    return corrected_string
# suggest title
def suggest_title():
    try:
        TItlelist = generate_title()
        corrected_word = spellCheck(' '.join(TItlelist))  
        avg_score, ci_score, avg_comments, ci_comments, avg_awards, ci_awards = get_predictions(corrected_word, tokenizer, max_text_length, score_model, comments_model, awards_model)
        print("title:", corrected_word)
        print(f"Predicted Score: {avg_score} (Confidence Interval: {ci_score})")
        print(f"Predicted Comments: {avg_comments} (Confidence Interval: {ci_comments})")
        print(f"Predicted Awards: {avg_awards} (Confidence Interval: {ci_awards})")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
# ------------------------------------------- image
categories = [worksheet.cell(row=i, column=7).value for i in range(2, worksheet.max_row + 1)]
image_urls = [worksheet.cell(row=i, column=6).value for i in range(2, worksheet.max_row + 1)]
# Define an image data generator for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,  # Random rotation up to 20 degrees
    width_shift_range=0.1,  # Random horizontal shift
    height_shift_range=0.1,  # Random vertical shift
    shear_range=0.2,  # Shearing transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill mode for empty spaces after augmentation
)
target_size = (224, 224)
# process image
def process_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img)  # Convert image to numpy array
            img_array = img_array / 255.0
            return img_array
    except Exception as e:
        print(f"Error processing image from URL {url}: {str(e)}")
# Process images concurrently
def get_prediction(image_url):
    # Process the input image concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_images = list(executor.map(process_image, [image_url]))
    
    if not processed_images or all(img is None for img in processed_images):
        return "Error: Unable to process the image."
    # Preprocess images using the data generator
    model = Sequential()
    preprocessed_images = datagen.flow(np.array(processed_images), batch_size=len(processed_images))
    predictions = model.predict(preprocessed_images)

    predicted_category_index = np.argmax(predictions, axis=1)

    if predicted_category_index.shape == (1,):
        predicted_category_index = predicted_category_index[0]
        predicted_category_index = int(predicted_category_index)
        predicted_category_name = categories[predicted_category_index]
        result = f"Predicted Category: {predicted_category_name}"
    else:
        result = "Error: Unable to predict category."
    return result
# -- main
if __name__ == '__main__':
    for _ in range(10):
        suggest_title()
        image_url = random.choice(NewImageList)
        prediction_result = get_prediction(image_url)
        print(prediction_result)
