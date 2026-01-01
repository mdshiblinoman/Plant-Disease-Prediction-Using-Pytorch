import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os

# Page configuration
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿",
    layout="centered"
)

# Model Definition (must match the trained model architecture)
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Input: 224x224 -> Conv1: 222x222 -> Pool1: 111x111 -> Conv2: 109x109 -> Pool2: 54x54
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 54 * 54, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load class indices
@st.cache_data
def load_class_indices():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # Convert string keys to int
    return {int(k): v for k, v in class_indices.items()}

# Load model
@st.cache_resource
def load_model(num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlantDiseaseCNN(num_classes)
    model.load_state_dict(torch.load('plant_disease_prediction_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Image preprocessing
def preprocess_image(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor

# Prediction function
def predict(model, image, class_indices, device):
    img_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_indices[predicted.item()]
    return predicted_class, confidence.item() * 100

# Main app
def main():
    st.title("🌿 Plant Disease Prediction")
    st.markdown("---")
    st.write("Upload an image of a plant leaf to detect diseases.")
    
    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application uses a Convolutional Neural Network (CNN) 
        to identify plant diseases from leaf images.
        
        **Supported Plants:**
        - Apple
        - Blueberry
        - Cherry
        - Corn
        - Grape
        - Orange
        - Peach
        - Pepper
        - Potato
        - Raspberry
        - Soybean
        - Squash
        - Strawberry
        - Tomato
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write("**How to use:**")
    st.sidebar.write("1. Upload a clear image of a plant leaf")
    st.sidebar.write("2. Wait for the model to analyze")
    st.sidebar.write("3. View the prediction results")
    
    # Check if model files exist
    if not os.path.exists('class_indices.json'):
        st.error("❌ class_indices.json file not found. Please ensure the file exists in the same directory.")
        return
    
    if not os.path.exists('plant_disease_prediction_model.pth'):
        st.error("❌ Model file not found. Please ensure 'plant_disease_prediction_model.pth' exists in the same directory.")
        return
    
    # Load class indices and model
    try:
        class_indices = load_class_indices()
        num_classes = len(class_indices)
        model, device = load_model(num_classes)
        
        st.sidebar.markdown("---")
        st.sidebar.success(f"✅ Model loaded successfully!")
        st.sidebar.write(f"📊 Number of classes: {num_classes}")
        st.sidebar.write(f"💻 Device: {device}")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a plant leaf"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)
        
        # Make prediction
        with st.spinner("🔍 Analyzing image..."):
            predicted_class, confidence = predict(model, image, class_indices, device)
        
        with col2:
            st.subheader("🔬 Prediction Results")
            
            # Parse prediction
            parts = predicted_class.split('___')
            plant_name = parts[0].replace('_', ' ')
            condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
            
            st.markdown(f"**Plant:** {plant_name}")
            st.markdown(f"**Condition:** {condition}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar for confidence
            st.progress(confidence / 100)
            
            # Health status indicator
            if 'healthy' in condition.lower():
                st.success("✅ The plant appears to be healthy!")
            else:
                st.warning(f"⚠️ Disease detected: {condition}")
        
        st.markdown("---")
        
        # Additional information
        with st.expander("📋 View All Class Names"):
            classes_list = list(class_indices.values())
            # Group by plant
            plants = {}
            for cls in classes_list:
                parts = cls.split('___')
                plant = parts[0].replace('_', ' ')
                condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
                if plant not in plants:
                    plants[plant] = []
                plants[plant].append(condition)
            
            for plant, conditions in sorted(plants.items()):
                st.write(f"**{plant}:** {', '.join(conditions)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Made with ❤️ using Streamlit and PyTorch</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
