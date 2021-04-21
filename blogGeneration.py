from transformers import GPT2Tokenizer, GPT2Model

#load model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2Model.from_pretrained('gpt2-medium',pad_token_id=tokenizer.eos_token_id)

#tokenization step
sentence = "Aristole was a great philosopher!"
input_id = tokenizer.encode(sentence, return_tensors='pt')

input_id
# tokenizer.decode(input_id[0])

#generate and decode text 
output_text = model.generate(input_id, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

#write output to text file
text = tokenizer.decode(output_text[0], skip_special_tokens=True)
with open('GeneratedText.txt', 'w') as f:
    f.write(text)