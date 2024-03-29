## IntroToNLP Project - QuestionAnswering (Spring '23)

#### Term Project for the course Introduction to Natural Language Processing for Spring'23 at IIIT, Hyderabad

#### Team : Rahul Padhy (2022201003) and Arun Das (2022201021)

#### SQuAD 1.0 and 2.0 datasets availed from [Kaggle](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset) and [HuggingFace](https://huggingface.co/datasets/squad_v2)

#### Kaggle Notebooks can be found [here](https://www.kaggle.com/jimhalpert26/code)

#### Pre-trained models on SQuAD 1.0 and SQuAD 2.0 can be found in the respective folders [here](https://drive.google.com/drive/folders/12kU8E_ti-F2cQ67sjRb5ymZl1K-234d3?usp=sharing)

#### Github Link for all the codes and experimentations can be found [here](https://github.com/imRP26/iNLP_QnA)

#### Method of Execution of the 2 scripts :- 

##### python3 TrainingBERT.py -t <train_dataset_path> -v <validation_dataset_path>

##### python3 FineTunedBERTevaluation.py -m <pre_trained_model_path> -v <validation_dataset_path>

##### NOTE : Its assumed that the required files are present in the same directory as the script(s).
