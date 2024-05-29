import tensorflow as tf
import time

model = tf.keras.models.load_model("food_vision_model.h5")
#m.predict()

class_names = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheesecake',
 'cheese_plate',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your TensorFlow model
#model = tf.keras.models.load_model('model1')

# Function to preprocess the image
def preprocess_image(image):
    # Resize image to desired dimensions
    image = image.resize((224, 224))
    # Convert image to numpy array
    img_array = np.array(image)
    #img = tf.image.resize(img_array,[224,224])
    #img = np.expand_dims(img_array, axis=0).astype(np.float32)
    # Normalize pixel values to range [0, 1]
    #img_array = img_array / 255.0
    # Add batch dimension and convert to float32
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# Streamlit app
def main():
    st.title("Image Prediction with TensorFlow Model")
    st.write("Upload an image and let the model predict the output.")

    # Image uploader
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        my_bar = st.progress(0,text='Progress Bar')
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        processed_image = preprocess_image(image)
        for percent_complete in range(50):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text='Preprocessing..')
        # Make prediction
        prediction = model.predict(processed_image)
        for percent_complete in range(50,100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text='Predicting..')

        # Display prediction
        st.write("Prediction Probabilities:", prediction)
        st.info(f"Prediction Label: {class_names[np.argmax(prediction)].capitalize()}")
        pred_conf = np.max(prediction)
        if pred_conf > 0.7:
            st.success(f'Prediction Confidence: {round(pred_conf*100,2)}%')
        else:
            st.error(f'Prediction Confidence: {round(pred_conf*100,2)}%')

if __name__ == "__main__":
    main()
