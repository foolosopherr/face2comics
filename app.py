import streamlit as st
from utils import load_checkpoint
from generator_model import Generator
import torch.optim as optim
from PIL import Image
import torch
from torchvision import transforms

DEVICE = 'cpu'
LEARNING_RATE = 2e-4
CHECKPOINT_GEN = "gen.pth.tar"

# @st.cache
def load_model():
    gen = Generator(in_channels=3).to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    checkpoint = torch.load(CHECKPOINT_GEN, map_location=DEVICE)
    gen.load_state_dict(checkpoint['state_dict'])
    opt_gen.load_state_dict(checkpoint['optimizer'])

    for param_group in opt_gen.param_groups:
        param_group['lr'] = LEARNING_RATE
    gen.eval()
    return gen 

model = load_model()

st.title('Face to comics style image transformation using pix2pix')
st.header('The model was trained for 50 epochs on 256x256 images')

tabs = ['Your image', 'Images from validation set', 'Images after every second epoch']
tab1, tab2, tab3 = st.tabs(tabs=tabs)

def transform_image(raw_image, model=model):
    transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
            )
        
    image = transformer(raw_image).view((1, 3, 256, 256))
    image = image.to(DEVICE)
    with torch.no_grad():
        image = model(image)
        image = image * 0.5 + 0.5
    image = transforms.ToPILImage()(image.view((3, 256, 256)))
    return image

with tab1:
    st.header('Crop your image into square shape to get the best result')
    raw_image = st.file_uploader('Upload your image')
    if raw_image:
        raw_image = Image.open(raw_image)
        image = transform_image(raw_image)

        col1, col2 = st.columns(2)
        with col1:
            st.header('Original')
            st.image(raw_image)
        with col2:
            st.header('Transformed')
            st.image(image)

with tab2:
    number = st.number_input('Choose image from validation set', 1, 250)
    number += 9499
    tab2_img = Image.open(f'val faces/{number}.jpg')
    tab2_img_tr = transform_image(tab2_img)
    col3, col4 = st.columns(2)
    with col3:
        st.header('Original')
        st.image(tab2_img.resize((256, 256)))
    with col4:
        st.header('Transformed')
        st.image(tab2_img_tr)

with tab3:
    number = st.number_input('Choose image from validation set', 1, 5)
    tab3_face = Image.open(f'content/evaluation/{number}/input_0.png')
    tab3_comics = Image.open(f'content/evaluation/{number}/label_0.png')
    tab3_concat = Image.open(f'content/evaluation/{number}.png')
    col5, col6 = st.columns(2)
    with col5:
        st.header('Original face')
        st.image(tab3_face)
    with col6:
        st.header('Perfect comics')
        st.image(tab3_comics)

    st.header('Model results after every second epoch')
    st.image(tab3_concat)



