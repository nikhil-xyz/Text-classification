import gradio as gr
import string
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

model = DistilBertForSequenceClassification.from_pretrained('./model/', problem_type="multi_label_classification", return_dict=True, num_labels=29)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
           'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
           'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
           'surprise', 'neutral', 'example_very_unclear']
def score_comment(comment):
    comment = comment.translate(str.maketrans('', '',string.punctuation))
    comment = comment.lower()
    comment = comment.split()
    comment = [wnl.lemmatize(word) for word in comment if not word in stopwords.words('english')]
    comment = ' '.join(comment)

    encoding = tokenizer(comment, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    outputs = model(**encoding)[0][0] > -1
    # print(outputs)
    text = ''
    for index, col in enumerate(columns):
        if outputs[index]:
            text += '{}\n'.format(col)
    return text

interface = gr.Interface(fn=score_comment,
                inputs=gr.inputs.Textbox(lines=2, placeholder='Express your feelings!'),
                outputs='text')
interface.launch(share=True)