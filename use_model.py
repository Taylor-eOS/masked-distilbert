import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from settings import TOKENIZER, MODEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MODEL.to(device)

def complete_masked_text(prompt):
    prompt = prompt.replace('MASK', TOKENIZER.mask_token)
    inputs = TOKENIZER.encode_plus(prompt, return_tensors='pt').to(device)
    mask_token_index = torch.where(inputs['input_ids'] == TOKENIZER.mask_token_id)[1]
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    for mask_index in mask_token_index:
        mask_logits = logits[0, mask_index, :]
        top_token_id = torch.argmax(mask_logits).item()
        predicted_token = TOKENIZER.decode([top_token_id]).strip()
        prompt = prompt.replace(TOKENIZER.mask_token, predicted_token, 1)
    return prompt

while True:
    user_input = input("Enter your prompt with 'MASK' (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    completion = complete_masked_text(user_input)
    print(f"Completed Text: {completion}")
