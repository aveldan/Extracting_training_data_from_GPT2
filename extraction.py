import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import math
import zlib
import numpy as np

def perplexity(text, model, tokenizer, compute_device, sliding=False):

    input = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input = input.to(compute_device)
    
    with torch.no_grad():
        output = model(input, labels=input)
    return torch.exp(output.loss)
    

def append_to_file(metric, generated_samples_unique, text, file_name="extracted_data.txt", n=10):

    
    file = open(file_name, 'a')
    idxs = np.argsort(metric)[::-1][:n]

    i = 1
    for idx in idxs:
        file.write("------->"+str(i)+": Generated sample\n")
        file.write(generated_samples_unique[idx])
        file.write("\n\n######"+text+" value= "+str(metric[idx])+"\n\n")
        i += 1

# Yet to add model_s, model_m
# and sliding perplexity over sliding window
def evaluating(generated_samples, model_xl, tokenizer, compute_device):

    perplexity_values = {"XL": [], "M": [], "S": [], "Lower": [],"zlib": [], "Window": []}

    generated_samples_unique = np.unique(generated_samples).tolist()

    print("Calculating perplexity scores...")

    for text in generated_samples_unique:

        perplexity_model_xl = perplexity(text, model_xl, tokenizer, compute_device)
        perplexity_values["XL"].append(perplexity_model_xl)
        
        perplexity_model_xl_lower = perplexity(text.lower(), model_xl, tokenizer, compute_device)
        perplexity_values["Lower"].append(perplexity_model_xl_lower)

        zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
        perplexity_values["zlib"].append(zlib_entropy)
    
    perplexity_values["XL"] = np.asarray(perplexity_values["XL"])
    perplexity_values["Lower"] = np.asarray(perplexity_values["Lower"])
    perplexity_values["zlib"] = np.asarray(perplexity_values["zlib"])
    
    print("Appending to the file...")
    # metric 1 (Perplexity of XL model)
    metric = -np.log(perplexity_values["XL"])
    append_to_file(metric, generated_samples_unique, "Perplexity of the XL model")

    # metric 2 (perplexity of XL on lower case/perplexity of XL on normal case)
    metric = np.log(perplexity_values["Lower"])/np.log(perplexity_values["XL"])
    append_to_file(metric, generated_samples, "perplexity of XL on lower case/perplexity of XL on normal case")

    # metric 3 (zlib entropy/perplexity of XL)
    metric = perplexity_values["zlib"]/np.log(perplexity_values["XL"])
    append_to_file(metric, generated_samples_unique, "zlib entropy/perplexity of XL")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)

    args = parser.parse_args()

    compute_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    model_xl = AutoModelForCausalLM.from_pretrained("gpt2-xl", low_cpu_mem_usage=True).to(compute_device)
    model_xl.config.pad_token_id = model_xl.config.eos_token_id
    model_xl.eval()

    num_batches = int(math.ceil(args.N / args.batch_size))

    generated_samples = []

    for batch in range(num_batches):
        
        prompts = [tokenizer.eos_token] * args.batch_size
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(compute_device)

        print("Generating sequences with max_length = 256 and top_k = 40...")
        generated_sequences = model_xl.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_length = 256,
            do_sample = True,
            top_k = 40,
            top_p = 1.0
        )

        generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        generated_samples += generated_texts
    
    evaluating(generated_samples, model_xl, tokenizer, compute_device)