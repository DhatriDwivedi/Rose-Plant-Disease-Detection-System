import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

def get_css():
    return """
    <style>
    body {
        background-color: #f0f4f7; /* Light background color */
        color: #333; /* Dark text color */
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
    }
    h1, h2, h3 {
        color: #4CAF50; /* Green color for headers */
        text-align: center; /* Centered headers */
    }
    .stButton > button {
        background-color: #4CAF50; /* Green button */
        color: white; /* White text */
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    @keyframes fall {
        0% { transform: translateY(0); }
        100% { transform: translateY(100vh); }
    }
    .falling-leaf {
        position: absolute;
        top: -10%;
        animation: fall 3s linear infinite;
        opacity: 0.8;
        z-index: 10;
    }
    .markdown-text {
        text-align: justify; /* Justify text for better readability */
    }
    </style>
    """

def display_leaves():
    leaves = ["🍃", "🍂", "🌿"]  # Use green leaf emojis
    leaves_html = "".join([f'<div class="falling-leaf" style="left: {np.random.randint(0, 100)}%; font-size: 30px;">{leaf}</div>' for leaf in leaves])
    return leaves_html

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "OIP.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ‘Plant-Pulse’ is an innovative web-based platform that addresses major agricultural concerns. To help users, the platform includes a plant disease prediction system as well as a direct sales marketplace.
    The disease prediction system uses convolutional neural networks (CNNs) to evaluate plant photos and identify probable health concerns early on, assisting farmers in mitigating damage and losses. 
    The direct sales option allows users to avoid mediators and sell their goods straight to merchants, increasing profit margins. Plant-Pulse's goal in merging these modern technologies is to increase agricultural output and financial security for users.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image is not None:
        st.image(test_image, width=400, use_column_width=True)

    # Predict button
    if st.button("Predict") and test_image is not None:
        # Inject CSS for falling leaves
        st.markdown(get_css(), unsafe_allow_html=True)
        leaves_html = display_leaves()
        st.markdown(leaves_html, unsafe_allow_html=True)  # Display falling leaves

        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = ['Healthy_Leaf_Rose', 'Rose_Rust', 'Rose_sawfly_Rose_slug']
        st.success("Model is predicting it's a {}".format(class_name[result_index]))
