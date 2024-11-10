import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = 'model'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForMaskedLM.from_pretrained(model_dir)
model.to(device)

def complete_masked_text(prompt):
    prompt = prompt.replace('MASK', tokenizer.mask_token)
    inputs = tokenizer.encode_plus(prompt, return_tensors='pt').to(device)
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    for mask_index in mask_token_index:
        mask_logits = logits[0, mask_index, :]
        top_token_id = torch.argmax(mask_logits).item()
        predicted_token = tokenizer.decode([top_token_id]).strip()
        prompt = prompt.replace(tokenizer.mask_token, predicted_token, 1)
    return prompt

while True:
    user_input = input("Enter your prompt with 'MASK' (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    completion = complete_masked_text(user_input)
    print(f"Completed Text: {completion}")

