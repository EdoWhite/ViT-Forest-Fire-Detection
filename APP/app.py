import streamlit as st
from transformers import pipeline as pip
from PIL import Image

# set page setting
st.set_page_config(page_title='Smoke & Fire Detection')

# set history var
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache(persist=True, allow_output_mutation=True)
def loadModel():
    pipeline = pip(task="image-classification", model="EdBianchi/vit-fire-detection")
    return pipeline

# PROCESSING
def compute(image):
    predictions = pipeline(image)

    with st.container():
        st.image(image, use_column_width=True)

    with st.container():
        st.write("### Classification Outputs:")
        col1, col2, col6 = st.columns(3)
        col1.metric(predictions[0]['label'], str(round(predictions[0]['score']*100, 1))+"%")
        col2.metric(predictions[1]['label'], str(round(predictions[1]['score']*100, 1))+"%")
        col6.metric(predictions[2]['label'], str(round(predictions[2]['score']*100, 1))+"%")
    return None

# INIT
with st.spinner('Loading the model, this could take some time...'):
    pipeline = loadModel()

# TITLE
st.write("# 🌲 Smoke and Fire in Forests 🌲")
st.write("""Wildfires or forest fires are **unpredictable catastrophic and destructive** events that affect **rural areas**. 
The impact of these events affects both **vegetation and wildlife**. 

This application showcases the **vit-fire-detection** model, a version of google **vit-base-patch16-224-in21k** vision transformer fine-tuned for **smoke and fire detection**. In particular, we can imagine a setup in which webcams, drones, or other recording devices **take pictures of a wild environment every t seconds or minutes**. The proposed system is then able to classify the current situation as **normal, smoke, or fire**. 
""")

st.write("### Upload an image to see the classifier in action")
# INPUT IMAGE
file_name = st.file_uploader("")
if file_name is not None:
    # USER IMAGE
    image = Image.open(file_name)
    compute(image)
else:
    # DEMO IMAGE
    demo_img = Image.open("./demo.jpg")
    compute(demo_img)

# SIDEBAR
st.sidebar.write("""
The fine-tuned model is hosted on the [Hugging Face Hub](https://huggingface.co/EdBianchi/vit-fire-detection).

The dataset for fine-tuning process was custom made from different datasets, in particular:

- Samples from "train_fire" and samples from "train_smoke" from [forest-fire dataset](https://www.kaggle.com/datasets/kutaykutlu/forest-fire?select=train_fire).
- All the samples (mixed together from further splitting) from [forest-fire-images dataset](https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images).

The custom dataset is hosted on the [Hugging Face Hub](https://huggingface.co/datasets/EdBianchi/SmokeFire).
""")