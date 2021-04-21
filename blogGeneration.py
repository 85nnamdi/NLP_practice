from transformers import GPT2LMHeadModel, GPT2Tokenizer

#load model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large',pad_token_id=tokenizer.eos_token_id)

#tokenization step
sentence = "Aristole was a great philosopher!"
input_id = tokenizer.encode(sentence, return_tensors='pt')

input_id
# tokenizer.decode(input_id[0])