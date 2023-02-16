import os
import json
from PIL import Image
import torch
import random
import numpy as np
from transformers import LayoutLMv2FeatureExtractor
from thefuzz import fuzz

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_data(config, mode):
    file = mode+'/'+mode+'_v1.0.json'
    with open(os.path.join(config.data_dir, file)) as f:
        data = json.load(f)

    return data

def get_extractor(config):
    if config.model == 'layoutlmv2':
        return LayoutLMv2FeatureExtractor()

def get_ocr_results_rearrange(examples, config, mode):
    ocr_root_dir = config.data_dir + "/" + mode + "/ocr_results/"
    image_root_dir = config.data_dir + "/" + mode + "/"
    ids = examples['ucsf_document_id']
    nums = examples['ucsf_document_page_no']

    images = [Image.open(image_root_dir + image_file).convert("RGB")
              for image_file in examples['image']]

    images = [image.resize(size=(224, 224), resample=Image.BILINEAR)
              for image in images]
    images = [np.array(image) for image in images]

    images = [image[::-1, :, :] for image in images]
    
    batch_words, batch_boxes = [], []
    for i in range(len(ids)):
        each_words, each_boxes = [], []
        path = ocr_root_dir + ids[i] + "_" + nums[i] + ".json"
        with open(path) as f:
            ocr = json.load(f)

        image_width, image_height = ocr['recognitionResults'][0]['width'], ocr['recognitionResults'][0]['height']
        lines: list[dict] = ocr['recognitionResults'][0]['lines']
        for line in lines:
            words: list[dict] = line['words']
            for word in words:
                boundingBox: list[int] = word['boundingBox']
                text: str = word['text']
                x1, y1, x2, y2, x3, y3, x4, y4 = boundingBox
                xs, ys = [x1, x2, x3, x4], [y1, y2, y3, y4]
                x_max, x_min, y_max, y_min = max(xs), min(xs), max(ys), min(ys)
                
                left, upper, right, lower = normalize_bbox(
                    [x_min, y_min, x_max, y_max], image_width, image_height)
                assert all(0 <= (each) <= 1000 for each in [
                    left, upper, right, lower])

                each_words.append(text)
                each_boxes.append([left, upper, right, lower])

        new_words = []
        new_boxes = []
        groups = []
        boundaries = []
        x_=config.x_
        y_=config.y_
        for word, box in zip(each_words, each_boxes):
            added = False
            for idx, boundary in enumerate(boundaries):
                gap_x=(box[2]-box[0])/len(word)*x_
                gap_y=(box[3]-box[1])*y_
                if boundary[0] - gap_x < box[0] < boundary[2] + gap_x and boundary[1] - gap_y < box[1] < boundary[3] + gap_y:
                    groups[idx].append((word, box))
                    added = True
                    boundaries[idx] = [min(box[0], boundary[0]), min(box[1], boundary[1]), max(box[2], boundary[2]), max(box[3], boundary[3])]
                    break
            if not added:
                boundaries.append(box)
                groups.append([(word, box)])

        for idx, group in enumerate(groups):
            for word, box in group:
                new_words.append(word)
                new_boxes.append(box)
        
        batch_words.append(new_words)
        batch_boxes.append(new_boxes)

    examples['image'] = images
    examples['words'] = batch_words
    examples['boxes'] = batch_boxes

    return examples

def get_ocr_results(examples, config, mode):
    ocr_root_dir = config.data_dir + "/" + mode + "/ocr_results/"
    image_root_dir = config.data_dir + "/" + mode + "/"
    ids = examples['ucsf_document_id']
    nums = examples['ucsf_document_page_no']

    images = [Image.open(image_root_dir + image_file).convert("RGB")
              for image_file in examples['image']]

    images = [image.resize(size=(224, 224), resample=Image.BILINEAR)
              for image in images]
    images = [np.array(image) for image in images]

    images = [image[::-1, :, :] for image in images]

    # text processing
    batch_words, batch_boxes = [], []
    for i in range(len(ids)):
        each_words, each_boxes = [], []
        path = ocr_root_dir + ids[i] + "_" + nums[i] + ".json"
        with open(path) as f:
            ocr = json.load(f)

        image_width, image_height = ocr['recognitionResults'][0]['width'], ocr['recognitionResults'][0]['height']
        lines: list[dict] = ocr['recognitionResults'][0]['lines']
        for line in lines:
            words: list[dict] = line['words']
            for word in words:
                boundingBox: list[int] = word['boundingBox']
                text: str = word['text']
                x1, y1, x2, y2, x3, y3, x4, y4 = boundingBox
                xs, ys = [x1, x2, x3, x4], [y1, y2, y3, y4]
                x_max, x_min, y_max, y_min = max(xs), min(xs), max(ys), min(ys)
                
                left, upper, right, lower = normalize_bbox(
                    [x_min, y_min, x_max, y_max], image_width, image_height)
                assert all(0 <= (each) <= 1000 for each in [
                    left, upper, right, lower])

                each_words.append(text)
                each_boxes.append([left, upper, right, lower])

        batch_words.append(each_words)
        batch_boxes.append(each_boxes)

    examples['image'] = images
    examples['words'] = batch_words
    examples['boxes'] = batch_boxes

    return examples

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def get_ocr_words_and_boxes(examples, config, feature_extractor, mode):

    root_dir = os.path.join(config['data_dir'], mode)
    # get a batch of document images
    images = [Image.open(root_dir + '/' + image_file).convert("RGB")
              for image_file in examples['image']]

    # resize every image to 224x224 + apply tesseract to get words + normalized boxes
    encoded_inputs = feature_extractor(images)

    examples['image'] = encoded_inputs.pixel_values
    examples['words'] = encoded_inputs.words
    examples['boxes'] = encoded_inputs.boxes

    return examples

def subfinder(words_list, answer_list):
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if len(answer_list) == 0:
            continue
        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0

def subfinder_similar(words_list, answer_list, question):
    matches = []
    start_indices = []
    end_indices = []
    question_set=set(question.split())
    sim=[]
    for idx, i in enumerate(range(len(words_list))):
        if len(answer_list) == 0:
            continue
        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            front=set(words_list[max(0,i-len(question_set)):i])
            back=set(words_list[i+len(answer_list):min(i+len(question_set)+len(answer_list),len(words_list))])
            front_recall=len(front&question_set)/(len(front)+1)
            back_recall=len(back&question_set)/(len(back)+1)
            sim.append(max(front_recall, back_recall))
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return sorted(zip(sim,matches))[-1][1], sorted(zip(sim,start_indices))[-1][1], sorted(zip(sim,end_indices))[-1][1]
    else:
        return None, 0, 0

def subfinder_multi(words_list, answer_list):
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if len(answer_list) == 0:
            continue
        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return matches, start_indices, end_indices
    else:
        return None, 0, 0

def fuzzy_matching(words_example, answer):
    step = len(answer)
    max_ratio = 0
    match, start_idx, end_idx = None, None, None
    for i in range(0, len(words_example)):
        target = words_example[i:i+step]
        ratio = fuzz.ratio(target, answer)
        if ratio > max_ratio:
            max_ratio = ratio
            match, start_idx, end_idx = target, i, i+(step-1)

    if max_ratio < 80:
        return None, None, None, None
    return match, start_idx, end_idx, max_ratio

def logging(file_name, text):
    if not os.path.exists(f'/opt/ml/experiments/final-project-level2-recsys-13/{file_name}'):
        with open(file_name, 'w') as f:
            f.write('0')

    with open(file_name, 'r') as f:
        first_line = int(f.readline())
        first_line += 1
        prev_text = f.read()
        
    with open(file_name, 'w') as f:
        f.write(str(first_line) + '\n')
        f.write(prev_text)
        f.write(text + '\n')


def encode_dataset(examples, tokenizer, mode='train', max_length=512):
    # take a batch
    questions = examples['question']
    words = examples['words']
    boxes = examples['boxes']

    # encode it
    encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)
    encoding['image'] = examples['image']
    
    # next, add start_positions and end_positions
    if mode == 'test':
        encoding['word_ids']        = [[-1 if id is None else id for id in encoding.word_ids(i)] 
                                       for i in range(len(examples['question']))]
        encoding['start_positions'] = [0] * len(examples['question'])
        encoding['end_positions']   = [0] * len(examples['question'])
        return encoding
    
    start_positions = []
    end_positions = []
    answers = examples['answers']
    # for every example in the batch:
    for batch_index in range(len(answers)):
        cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
        # try to find one of the answers in the context, return first match
        words_example = [word.lower() for word in words[batch_index]]
        for answer in answers[batch_index]:
            match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())
            if match:
                break
        # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
        if not match:
            max_ratio = 0
            for answer in answers[batch_index]:
                curr_match, curr_word_idx_start, curr_word_idx_end, curr_ratio = fuzzy_matching(words_example, answer.lower().split())
                if curr_match and curr_ratio > max_ratio:
                    max_ratio = curr_ratio
                    match, word_idx_start, word_idx_end = curr_match, curr_word_idx_start, curr_word_idx_end
            # for logging
            if match:
                formatted_string = f"answer: {answers[batch_index]}| match: {match}| word_idx_start: {word_idx_start}| word_idx_end: {word_idx_end}| max_ratio: {max_ratio}| question: | {questions[batch_index]} | ref: {examples['ucsf_document_id'][batch_index]}_{examples['ucsf_document_page_no'][batch_index]}"
                logging('log_fuzzy_matching.txt', formatted_string)
        # END OF EXPERIMENT

        if match:
            sequence_ids = encoding.sequence_ids(batch_index)
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(encoding.input_ids[batch_index]) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            word_ids = encoding.word_ids(batch_index)[token_start_index:token_end_index+1]
            start_found, end_found = False, False
            for id in word_ids:
                if id == word_idx_start:
                    start_positions.append(token_start_index)
                    start_found = True
                    break
                else:
                    token_start_index += 1

            for id in word_ids[::-1]:
                if id == word_idx_end:
                    end_positions.append(token_end_index)
                    end_found = True
                    break
                else:
                    token_end_index -= 1

            # start position은 추가되었는데 end position은 추가되지 않은 경우
            if start_found:
                if not end_found:
                    end_positions.append(token_start_index)
            # start position도 추가되지 않은 경우
            else:
                match = False

        if not match:
            start_positions.append(cls_index)
            end_positions.append(cls_index)

    encoding['start_positions'] = start_positions
    encoding['end_positions']   = end_positions

    return encoding


def encode_with_stride(examples, tokenizer, mode='train', max_length=512, doc_stride=256):
    # take a batch
    questions = examples['question']
    words = examples['words']
    boxes = examples['boxes']

    # encode it
    encoding = tokenizer(
        questions, # examples["question"],
        words, # examples["context"],
        boxes,
        truncation="only_second",  # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True, # 길이를 넘어가는 토큰들을 반환할 것인지
        padding="max_length",
    )
    
    # next, add start_positions and end_positions
    if mode == 'test':
        encoding['word_ids']        = [[-1 if id is None else id for id in encoding.word_ids(i)] 
                                       for i in range(len(examples['question']))]
        encoding['start_positions'] = [0] * len(examples['question'])
        encoding['end_positions']   = [0] * len(examples['question'])
        return encoding
    
    start_positions = []
    end_positions = []
    image = []
    answers = examples['answers']
    
    # example 하나가 여러 sequence에 대응하는 경우를 위해 매핑이 필요함.
    overflow_to_sample_mapping = encoding.pop("overflow_to_sample_mapping")
    
    for i, offsets in enumerate(overflow_to_sample_mapping):
        input_ids = encoding['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        words_example = [word.lower() for word in words[offsets]]
        for answer in answers[offsets]:
            match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())
            if match:
                break
        
        if match:
            # 해당 example에 해당하는 sequence 찾기
            sequence_ids = encoding.sequence_ids(i)
            
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = max_length - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            word_ids = encoding.word_ids(i)[token_start_index:token_end_index+1]
            match_start = False
            for id in word_ids:
                if id == word_idx_start:
                    match_start = True
                    break
                else:
                    token_start_index += 1
            
            match_end = False
            for id in word_ids[::-1]:
                if id == word_idx_end:
                    match_end = True
                    break
                else:
                    token_end_index -= 1
            
            if not match_start or not match_end:
                match = False
            
        if match:
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
            image.append(examples['image'][offsets])
        
        else:
            start_positions.append(cls_index)
            end_positions.append(cls_index)    
            image.append(examples['image'][offsets])
    
    encoding['start_positions'] = start_positions
    encoding['end_positions']   = end_positions
    encoding['image'] = image
    
    return encoding


def predict(outputs):
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    predicted_start_idx_list = []
    predicted_end_idx_list = []

    # TODO vectorized code로 바꾸기 
    for i in range(len(start_logits)):
        predicted_start_idx = 0
        predicted_end_idx = 0
        max_score = -float('inf')
        for start in range(len(start_logits[i])):
            for end in range(start, len(end_logits[i])):
                score = start_logits[i][start] + end_logits[i][end]
                if score > max_score:
                    max_score = score
                    predicted_start_idx = start
                    predicted_end_idx = end
        predicted_start_idx_list.append(predicted_start_idx)
        predicted_end_idx_list.append(predicted_end_idx)
    
    
    
    return predicted_start_idx_list, predicted_end_idx_list


def predict_start_first(outputs):
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    predicted_start_idx_list = []
    predicted_end_idx_list = []
    
    start_position = start_logits.argmax(1)

    # TODO vectorized code로 바꾸기 
    for i in range(len(start_logits)):
        
        start = start_position[i]
        predicted_start_idx_list.append(start)
        max_score = -float('inf')
        predicted_end_idx = 0
        
        for end in range(start, len(end_logits[i])):
            score = end_logits[i][end]
            if score > max_score:
                max_score = score
                predicted_end_idx = end
                
        predicted_end_idx_list.append(predicted_end_idx)
    
    return predicted_start_idx_list, predicted_end_idx_list


def load_textract_result(examples, config, mode):
    ids = examples['ucsf_document_id']
    nums = examples['ucsf_document_page_no']
    files = [id + '_' + num + '.json' for id, num in zip(ids, nums)]
    
    batch_words, batch_boxes = [], []
    for file in files:
        each_words, each_boxes = [], []
        path = os.path.join(config.ocr_path, mode, file)
        with open(path) as f:
            ocr = json.load(f)
        
        for block in ocr['Blocks']:
            if block['BlockType'] != 'WORD':
                continue
            
            each_words.append(block['Text'])
            x, y = [], []
            for bbox in block['Geometry']['Polygon']:
                x.append(int(bbox['X'] * 1000))
                y.append(int(bbox['Y'] * 1000))
            each_boxes.append([min(x), min(y), max(x), max(y)])
    
        batch_words.append(each_words)
        batch_boxes.append(each_boxes)
    
    image_root_dir = config.data_dir + "/" + mode + "/"
    images = [Image.open(image_root_dir + image_file).convert("RGB")
              for image_file in examples['image']]
    images = [image.resize(size=(224, 224), resample=Image.BILINEAR)
              for image in images]
    images = [np.array(image) for image in images]
    images = [image[::-1, :, :] for image in images]
    
    examples['image'] = images
    examples['words'] = batch_words
    examples['boxes'] = batch_boxes

    return examples