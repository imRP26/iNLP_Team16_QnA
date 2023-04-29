import argparse
import json
from pathlib import Path
import string
import re
import torch
from transformers import AutoTokenizer, BertTokenizerFast, BertForQuestionAnswering, DistilBertTokenizer, \
                         DistilBertForQuestionAnswering, pipeline


def predict(context, query, model):
    inputs = tokenizer.encode_plus(query, context, return_tensors='pt')
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs[0])
    answer_end = torch.argmax(outputs[1]) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(\
                                                inputs['input_ids'][0][answer_start : answer_end]))
    return answer


'''
Removing articles and punctuation and standardizing whitespace
'''
def normalize_text(s):

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
        
    def white_space_fix(text):
        return ' '.join(text.split())
        
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
        
    def lower(text):
        return text.lower()
        
    return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    
def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))
    

def compute_f1(prediction,truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    # if either the prediction or the truth is no-answer, then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)
    
    
def give_an_answer(context, query, answer, model):
    prediction = predict(context, query)
    em_score = compute_exact_match(prediction, answer)
    f1_score = compute_f1(prediction, answer)
    print (f'Question : {query}')
    print (f'Prediction : {prediction}')
    print (f'True Answer : {answer}')
    print (f'EM : {em_score}')
    print (f'F1 : {f1_score}')
    return f1_score


def evaluation1(model):
    context = "Hi! My name is Alexa and I am 21 years old. I used to live in Peristeri of Athens, \
               but now I moved on in Kaisariani of Athens."
    queries = ["How old is Alexa?", 
               "Where does Alexa live now?", 
               "Where Alexa used to live?"]
    answers = ["21", 
               "Kaisariani of Athens", 
               "Peristeri of Athens"]
    for q, a in zip(queries, answers):
        give_an_answer(context, q, a, model)


def evaluation2(model):
    context = """Queen are a British rock band formed in London in 1970. Their classic line-up was 
                 Freddie Mercury (lead vocals, piano), Brian May (guitar, vocals), Roger Taylor 
                 (drums, vocals) and John Deacon (bass). Their earliest works were influenced by 
                 progressive rock, hard rock and heavy metal, but the band gradually ventured into 
                 more conventional and radio-friendly works by incorporating further styles, such as 
                 arena rock and pop rock."""
    queries = ["When was Queen found?", 
               "Who were the classic members of Queen band?", 
               "What kind of band they are?"]
    answers = ["1970", 
               "Freddie Mercury, Brian May, Roger Taylor and John Deacon", 
               "rock"]
    for q, a in zip(queries, answers):
        give_an_answer(context, q, a, model)


def evaluation3(model):
    context = """Mount Olympus is the highest mountain in Greece. It is part of the Olympus massif near 
                 the Gulf of Thérmai of the Aegean Sea, located in the Olympus Range on the border 
                 between Thessaly and Macedonia, between the regional units of Pieria and Larissa, 
                 about 80 km (50 mi) southwest from Thessaloniki. Mount Olympus has 52 peaks and deep 
                 gorges. The highest peak, Mytikas, meaning "nose", rises to 2917 metres (9,570 ft). 
                 It is one of the highest peaks in Europe in terms of topographic prominence."""
    queries = ["How many metres high is Mount Olympus?", 
               "What famous landmarks are near Mount Olympus?",
               "How far away is Olympus from Thessaloniki?"]
    answers = ["2917", 
               "Gulf of Thérmai of the Aegean Sea", 
               "80 km (50 mi)"]
    for q, a in zip(queries, answers):
        give_an_answer(context, q, a, model)


def evaluation4(model):
    context = """The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing pandemic 
                 of coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome 
                 coronavirus 2 (SARS-CoV-2). It was first identified in December 2019 in Wuhan, China. 
                 The World Health Organization declared the outbreak a Public Health Emergency of 
                 International Concern in January 2020 and a pandemic in March 2020. As of 6 February 
                 2021, more than 105 million cases have been confirmed, with more than 2.3 million deaths 
                 attributed to COVID-19. Symptoms of COVID-19 are highly variable, ranging from none to 
                 severe illness. The virus spreads mainly through the air when people are near each 
                 other.[b] It leaves an infected person as they breathe, cough, sneeze, or speak and 
                 enters another person via their mouth, nose, or eyes. It may also spread via 
                 contaminated surfaces. People remain infectious for up to two weeks, and can spread 
                 the virus even if they do not show symptoms.[9]"""
    queries = ["What is COVID-19?",
               "What is caused by COVID-19?",
               "How many cases have been confirmed from COVID-19?",
               "How many deaths have been confirmed from COVID-19?",
               "How is COVID-19 spread?",
               "How long can an infected person remain infected?",
               "Can a infected person spread the virus even if they don't have symptoms?",
               "What do elephants eat?"]
    answers = ["an ongoing pandemic of coronavirus disease 2019",
               "severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2)",
               "more than 105 million cases", 
               "more than 2.3 million deaths",
               "mainly through the air when people are near each other. It leaves an infected person as they breathe, cough, sneeze, or speak and enters another person via their mouth, nose, or eyes. It may also spread via contaminated surfaces.", 
               "up to two weeks", 
               "yes", 
               ""]
    for q, a in zip(queries, answers):
        give_an_answer(context, q, a, model)


def evaluation5(model):
    context = """Harry Potter is a series of seven fantasy novels written by British author, J. K. Rowling. 
                 The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and 
                 Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. 
                 The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who 
                 intends to become immortal, overthrow the wizard governing body known as the Ministry of 
                 Magic and subjugate all wizards and Muggles (non-magical people). Since the release of 
                 the first novel, Harry Potter and the Philosopher's Stone, on 26 June 1997, the books 
                 have found immense popularity, positive reviews, and commercial success worldwide. They 
                 have attracted a wide adult audience as well as younger readers and are often considered 
                 cornerstones of modern young adult literature.[2] As of February 2018, the books have 
                 sold more than 500 million copies worldwide, making them the best-selling book series in 
                 history, and have been translated into eighty languages.[3] The last four books 
                 consecutively set records as the fastest-selling books in history, with the final 
                 installment selling roughly eleven million copies in the United States within twenty-four 
                 hours of its release. """
    queries = ["Who wrote Harry Potter's novels?",
               "Who are Harry Potter's friends?",
               "Who is the enemy of Harry Potter?",
               "What are Muggles?",
               "Which is the name of Harry Poter's first novel?",
               "When did the first novel release?",
               "Who was attracted by Harry Potter novels?",
               "How many languages Harry Potter has been translated into? "]
    answers = ["J. K. Rowling",
               "Hermione Granger and Ron Weasley",
               "Lord Voldemort",
               "non-magical people",
               "Harry Potter and the Philosopher's Stone",
               "26 June 1997",
               "a wide adult audience as well as younger readers",
               "eighty"]
    for q, a in zip(queries, answers):
        give_an_answer(context, q, a, model)


'''
Retrieval and Storage of the Data
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
Evaluating performance on the validation / unseen dataset
'''
def evaluate_validation_data_performance(contexts, queries, answers, model, tokenizer):
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    avg_f1_score, avg_score, i = 0.0, 0.0, 0
    for context, question, answer in zip(contexts, queries, answers):
        print ('Query Number', i)
        try:
            f1 = give_an_answer(context, question, answer)
            avg_f1_score += f1
            i += 1
            result = qa_pipeline(question=question, context=context)
            score = result['score']
            print ('Pipeline Score :', score, '\n')
            avg_score += score
        except Exception as e:
            continue
    avg_f1_score /= i
    avg_score /= i
    print ('Mean of all the F1-Scores :', avg_f1_score)
    print ('Mean of all the Scores :', avg_score)


'''
The controller function of the script
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modelpath', dest='model_path', type=str, help='Path for fine-tuned BERT model.')
    parser.add_argument('-v', '--validationpath', dest='validation_path', type=str, help='Path for the validation dataset.')
    args = parser.parse_args()
    tokenizer1 = AutoTokenizer.from_pretrained('bert-base-uncased')
    model1 = torch.load(model_path, map_location=torch.device('cpu'))
    model1.eval()
    tokenizer2 = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model2 = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model2.eval()
    tokenizer3 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
    model3 = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
    model3.eval()
    # Simplest custom context - Hello Alexa!
    evaluation1(model1)
    evaluation1(model2)
    evaluation1(model3)
    # Context based upon a British Rock Band
    evaluation2(model1)
    evaluation2(model2)
    evaluation2(model3)
    # Context having numerical answers
    evaluation3(model1)
    evaluation3(model2)
    evaluation3(model3)
    # Context of COVID-19
    evaluation4(model1)
    evaluation4(model2)
    evaluation4(model3)
    # Context of Harry Porter
    evaluation5(model1)
    evaluation5(model2)
    evaluation5(model3)
    validation_path = Path(validation_path)
    validation_texts, validation_queries, validation_answers = generate_texts_queries_answers(validation_path)
    unique_queries = set()
    unique_validation_contexts, unique_validation_queries, unique_validation_answers = [], [], []
    for i in range(len(validation_queries)):
        if validation_queries[i] in unique_queries:
            continue
        unique_queries.add(validation_queries[i])
        unique_validation_contexts.append(validation_texts[i])
        unique_validation_queries.append(validation_queries[i])
        unique_validation_answers.append(validation_answers[i]['text'])
    evaluate_validation_data_performance(unique_validation_contexts, unique_validation_queries, \
                                         unique_validation_answers, model1, tokenizer1)
    evaluate_validation_data_performance(unique_validation_contexts, unique_validation_queries, \
                                         unique_validation_answers, model2, tokenizer2)
    evaluate_validation_data_performance(unique_validation_contexts, unique_validation_queries, \
                                         unique_validation_answers, model3, tokenizer3)


if __name__ == '__main__':
    main()
