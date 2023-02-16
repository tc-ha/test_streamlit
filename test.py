import streamlit as st
import torch
from PIL import Image
import json
from tqdm import tqdm

import hydra
from transformers import AutoModelForQuestionAnswering
from data_loader.data_loaders import DataLoader
from utils.util import predict_start_first

class Config():
    def __init__(self):
        self.data_dir = "/opt/ml/input/data/"
        self.model = "layoutlmv2"
        self.device = "cpu"
        self.checkpoint = "microsoft/layoutlmv2-base-uncased"
        self.use_ocr_library = False
        self.debug = False
        self.batch_data = 1
        self.num_proc = 1
        self.shuffle = True
        
        self.lr = 5e-6
        self.seed = 42
        self.batch = 1
        self.max_len = 512
        self.epochs = 1000
        
        self.fuzzy = False
        self.model_name = ''
        
config = Config()

# Define function to make predictions
def predict(config, image, question):
    
    model = AutoModelForQuestionAnswering.from_pretrained('microsoft/layoutlmv2-base-uncased', from_pt=True).to(config.device)

    data_loader = DataLoader(config, 'test')
    tokenizer = data_loader.tokenizer
    answers = []
    for idx, batch in enumerate(tqdm(data_loader.test_data_loader)):
        input_ids = batch["input_ids"].to(config.device)
        word_ids = batch['word_ids'].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        token_type_ids = batch["token_type_ids"].to(config.device)
        bbox = batch["bbox"].to(config.device)
        image = batch["image"].to(config.device)

        # forward + backward + optimize

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                bbox=bbox, image=image)
        
        predicted_start_idx, predicted_end_idx = predict_start_first(outputs)
        
        for batch_idx in range(batch['input_ids'].shape[0]):
            answer     = ""
            pred_start = predicted_start_idx[batch_idx]
            pred_end   = predicted_end_idx[batch_idx]
            word_id    = word_ids[batch_idx, pred_start]
            for i in range(pred_start, pred_end + 1):
                if word_id == word_ids[batch_idx, i]:
                    answer += tokenizer.decode(batch['input_ids'][batch_idx][i])
                else:
                    answer += ' ' + tokenizer.decode(batch['input_ids'][batch_idx][i])
                    word_id = word_ids[batch_idx, i]

            answer = answer.replace('##', '')

            answers.append(answer)

    ret = [{'questionId': qid, 'answer': answer} for qid,answer in zip(data_loader.test_df['questionId'].tolist(), answers)]
    return ret

def main(config):
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Load deep learning model
    checkpoint = ''
    # model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
    # model.load_state_dict(torch.load("model")) 

    # Create Streamlit app
    st.title('Deep Learning Pipeline')
    st.write('Upload an image and ask a question to get a prediction')

    # Create file uploader and text input widgets
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    question = st.text_input('Ask a question')

    # If file is uploaded, show the image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # If question is asked and file is uploaded, make a prediction
    if st.button('Get Prediction') and uploaded_file is not None and question != '':
        # Preprocess the image and question as needed
        # ...

        # Make the prediction
        with st.spinner('Predicting...'):
            output = predict(config, image, question)

        # Show the output
        st.write('Output:', output)


if __name__ == '__main__':
    config = Config()
    main(config)


    


