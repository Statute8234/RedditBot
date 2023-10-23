import openpyxl
import concurrent.futures
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
# path
path = r"C:\Users\danie\OneDrive\Desktop\Data.xlsx"
workbook = openpyxl.load_workbook(path)
worksheet = workbook.active
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
# Example usage:
image_url = "https://i.redd.it/blhc7mss0dvb1.png"
prediction_result = get_prediction(image_url)
print(prediction_result)
