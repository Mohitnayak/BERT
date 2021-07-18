import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import logging
logging.basicConfig(level=logging.INFO)

logging.info("Applying BERT directly")


#Model
logging.info("Loading Model")
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

logging.info("Model Load Complete")

#Tokenizer
logging.info("Loading tokenizer")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
logging.info("Tokenizer load complete")

# question = '''Why was the student group called "the Methodists?"'''

# paragraph = ''' The movement which would become The United Methodist Church began in the mid-18th century within the Church of England.
#             A small group of students, including John Wesley, Charles Wesley and George Whitefield, met on the Oxford University campus.
#             They focused on Bible study, methodical study of scripture and living a holy life.
#             Other students mocked them, saying they were the "Holy Club" and "the Methodists", being methodical and exceptionally detailed in their Bible study, opinions and disciplined lifestyle.
#             Eventually, the so-called Methodists started individual societies or classes for members of the Church of England who wanted to live a more religious life. '''

question = '''where do sedimentary rocks get its name from the fact that?'''
paragraph = '''Sedimentary rocks are formed by the process of sedimentation. Layer after layer of minerals is deposited over a great span of time, resulting in the formation of a sedimentary rock. As a result, each layer is different if the conditions under which its deposits were different. Thus we can say that a sedimentary rock is a sort of museum,
 holding the records of all the time over which it was formed, which by all means can be as long as a billion years.'''




logging.info("encoding started")
encoding = tokenizer.encode_plus(text=question, text_pair=paragraph,return_attention_mask = True)
logging.info("encoding ended")

inputs = encoding['input_ids']  # Token embeddings

sentence_embedding = encoding['token_type_ids']  # Segment embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs)


start_score = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))["start_logits"]
end_score = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))["end_logits"]
start_index = torch.argmax(start_score)

end_index = torch.argmax(end_score)

answer = ' '.join(tokens[start_index:end_index+1])

corrected_answer = ''

for word in answer.split():

    # If it's a subword token
    if word[0:2] == '##':
        corrected_answer += word[2:]
    else:
        corrected_answer += ' ' + word

print(corrected_answer)
