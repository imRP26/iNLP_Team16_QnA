import argparse
import json
from matplotlib import pyplot as plt
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, BertForQuestionAnswering


'''
In order to train and validate previous data more easily and covert encodings to datasets
'''
class SQuADdataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key : torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


'''
Retrieval and Storage of useful data from raw data
'''
def generate_texts_queries_answers(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    texts, queries, answers = [], [], []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    texts.append(context)
                    queries.append(question)
                    answers.append(answer)
    return texts, queries, answers


'''
Fixing up end position characters in train and validation data - Processing the SQuAD dataset to keep up with 
the input that BERT desires!
BERT Models need both the start and end position characters of the answer and sometimes its been noticed that 
SQuAD answers "eat up" 1 or 2 characters from the real answer in the passage
'''
def fix_end_position_chars(answers, texts):
    for answer, text in zip(answers, texts):
        real_answer = answer['text']
        answer_start_index = answer['answer_start']
        answer_end_index = answer_start_index + len(real_answer)
        if text[answer_start_index : answer_end_index] == real_answer:
            answer['answer_end'] = answer_end_index
        elif text[answer_start_index - 1 : answer_end_index - 1] == real_answer:
            answer['answer_start'] = answer_start_index - 1
            answer['answer_end'] = answer_end_index - 1
        elif text[answer_start_index - 2 : answer_end_index - 2] == real_answer:
            answer['answer_start'] = answer_start_index - 2
            answer['answer_end'] = answer_end_index - 2


'''
Conversion of the start-end positions to the tokens' start-end positions
'''
def add_token_positions(encodings, answers):
    start_positions, end_positions = [], []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
        # if start position is None, then it means that the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # if end position is None, the 'char_to_token' function points to the space after the correct token
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
            # if end position is still None, the answer passage has been truncated
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions' : start_positions, 'end_positions' : end_positions})


'''
Training the Model and Plotting the Train and Validation Losses
'''
def model_training_and_plotting(device, model, optimizer, epochs, train_loader, validation_loader):
    train_losses, validation_losses = [], []
    print_every = 1000
    for epoch in range(epochs):
        epoch_time = time.time()
        # Set the model in train mode
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, \
                            end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (batch_idx + 1) % print_every == 0:
                print ('Batch {:} / {:}'.format(batch_idx + 1, len(train_loader)), '\nLoss :', round(loss.item(), 1), '\n')
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        # Set the model in evaluation mode
        model.eval()
        epoch_loss = 0
        for batch_idx, batch in enumerate(validation_loader):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, \
                                end_positions=end_positions)
                loss = outputs[0]
                epoch_loss += loss.item()
            if (batch_idx + 1) % print_every == 0:
                print ('Batch {:} / {:}'.format(batch_idx + 1, len(validation_loader)), '\nLoss :', round(loss.item(), 1), '\n')
        epoch_loss /= len(validation_loader)
        validation_losses.append(epoch_loss)
        print ('\n---Epoch', epoch + 1, '---', '\nTraining Loss :', train_losses[-1], '---', '\nValidation Loss :', \
               validation_losses[-1], '\n---\n')
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title('Train and Validation Losses', size=20)
    ax.set_ylabel('Loss', fontsize=20)
    ax.set_xlabel('Epochs', fontsize=25)
    _ = ax.plot(train_losses)
    _ = ax.plot(validation_losses)
    _ = ax.legend(('Train', 'Validation'), loc='upper right')
    torch.save(model, 'bert_model_3_epochs.pt')


'''
Controller function of the script
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainpath', dest='train_path', type=str, help='Path for the train dataset.')
    parser.add_argument('-v', '--validationpath', dest='validation_path', type=str, help='Path for the validation dataset.')
    args = parser.parse_args()
    train_path = Path(train_path)
    train_texts, train_queries, train_answers = generate_texts_queries_answers(train_path)
    validation_path = Path(validation_path)
    validation_texts, validation_queries, validation_answers = generate_texts_queries_answers(validation_path)
    fix_end_position_chars(train_answers, train_texts)
    fix_end_position_chars(validation_answers, validation_texts)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, train_queries, truncation=True, padding=True)
    validation_encodings = tokenizer(validation_texts, validation_queries, truncation=True, padding=True)
    add_token_positions(train_encodings, train_answers)
    add_token_positions(validation_encodings, validation_answers)
    train_dataset = SQuADdataset(train_encodings)
    validation_dataset = SQuADdataset(validation_encodings)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    model_training_and_plotting(device, model, optimizer, epochs, train_loader, validation_loader)


if __name__ == '__main__':
    main()
