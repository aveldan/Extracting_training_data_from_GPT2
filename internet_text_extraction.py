import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import math
import zlib
import numpy as np


def perplexity(text, model, tokenizer, compute_device, sliding=False):

    input = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input = input.to(compute_device)
    
    if not sliding:

        with torch.no_grad():
            output = model(input, labels=input)
        return torch.exp(output.loss)
    
    else:

        min_perplexity = math.inf
        with torch.no_grad():
            for i in range(input.shape[0]-50):
                inp = input[i:i+50]
                output = model(inp, labels=inp)
                min_perplexity = min(min_perplexity, torch.exp(output.loss))
        
        return min_perplexity

def append_to_file(metric, generated_samples_unique, text, n, file_name):

    file_path = "./internet_text_output/"+file_name

    idxs = np.argsort(metric)[::-1][:n]

    file = open(file_path, 'w')
    i = 1
    for idx in idxs:
        file.write("------->"+str(i)+"Metric("+text+"): "+str(metric[idx])+"\n\n\n")
        file.write(generated_samples_unique[idx])
        file.write("\n\n")
        i += 1

def evaluating(generated_samples, model_s, model_m, model_xl, tokenizer, compute_device):

    perplexity_values = {"XL": [], "M": [], "S": [], "Lower": [],"zlib": [], "Window": []}

    generated_samples_unique = np.unique(generated_samples).tolist()

    print("Calculating perplexity scores...")

    for text in generated_samples_unique:
        perplexity_model_s = perplexity(text, model_s, tokenizer, compute_device)
        perplexity_values["S"].append(perplexity_model_s)

        perplexity_model_m = perplexity(text, model_m, tokenizer, compute_device)
        perplexity_values["M"].append(perplexity_model_m)

        perplexity_model_xl = perplexity(text, model_xl, tokenizer, compute_device)
        perplexity_values["XL"].append(perplexity_model_xl)
        
        perplexity_model_xl_lower = perplexity(text.lower(), model_xl, tokenizer, compute_device)
        perplexity_values["Lower"].append(perplexity_model_xl_lower)

        zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
        perplexity_values["zlib"].append(zlib_entropy)

        perplexity_model_xl_window = perplexity(text, model_xl, tokenizer, compute_device, sliding=True)
        perplexity_values["Window"].append(perplexity_model_xl_window)
    
    perplexity_values["S"] = np.asarray(perplexity_values["S"])
    perplexity_values["M"] = np.asarray(perplexity_values["M"])
    perplexity_values["XL"] = np.asarray(perplexity_values["XL"])
    perplexity_values["Lower"] = np.asarray(perplexity_values["Lower"])
    perplexity_values["zlib"] = np.asarray(perplexity_values["zlib"])
    perplexity_values["Window"] = np.asarray(perplexity_values["Window"])

    print("Appending to the file...")
    # metric 1 (Perplexity of XL model)
    metric = -np.log(perplexity_values["XL"])
    append_to_file(metric, generated_samples_unique, "Perplexity of the XL model", 100, "perplexity.txt")

    # metric 2 (perplexity of S/perplexity of XL)
    metric = np.log(perplexity_values["S"])/np.log(perplexity_values["XL"])
    append_to_file(metric, generated_samples, "Perplexity of the S model/Perplexity of XL model", 100, "small.txt")

    # metric 3 (perplexity of M/perplexity of XL)
    metric = np.log(perplexity_values["M"])/np.log(perplexity_values["XL"])
    append_to_file(metric, generated_samples, "Perplexity of the M model/Perplexity of XL model", 100, "medium.txt")

    # metric 4 (perplexity of XL on lower case/perplexity of XL on normal case)
    metric = np.log(perplexity_values["Lower"])/np.log(perplexity_values["XL"])
    append_to_file(metric, generated_samples, "Perplexity of XL on lower case/Perplexity of XL on normal case", 100, "lower.txt")

    # metric 5 (zlib entropy/perplexity of XL)
    metric = perplexity_values["zlib"]/np.log(perplexity_values["XL"])
    append_to_file(metric, generated_samples_unique, "zlib entropy/perplexity of XL", 100, "zlib.txt")

    # metric 6 (Minimum perplexity over a sliding window of size 50)
    metric = -np.log(perplexity_values["Window"])
    append_to_file(metric, generated_samples_unique,"Minimum perplexity over a sliding window of size 50", 100, "window.txt")


def parse_wet_files(file):

    with open(file) as f:
        lines = f.readlines()

    start_idx = []
    for i in range(len(lines)):
        if "WARC/1.0" in lines[i]:
            start_idx.append(i)
    
    plain_text = ""

    for i in range(len(start_idx)-1):
        start = start_idx[i]
        end = start_idx[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            for j in range(start+10, end):
                plain_text += lines[j]
    
    return plain_text

# Older common crawl files had a slightly different structure.
def parse_old_wet_files(file):

    with open(file) as f:
        lines = f.readlines()

    start_idx = []
    for i in range(len(lines)):
        if "WARC/1.0" in lines[i]:
            start_idx.append(i)
    
    plain_text = ""

    for i in range(len(start_idx)-1):
        start = start_idx[i]
        end = start_idx[i+1]
        tmp_plain_text = ""

        for j in range(start+10,end):
            tmp_plain_text += lines[j]

        if(tmp_plain_text.isascii()):
            plain_text += tmp_plain_text
    
    return plain_text

def generate_inputs_internet_text(wet_plain_text, batch_size):

    input_ids = []
    attention_mask = []

    while len(input_ids) < batch_size:
        r = np.random.randint(0, len(wet_plain_text))
        prompt = " ".join(wet_plain_text[r:r+100].split(" ")[1:-1])

        inputs = tokenizer(prompt, return_tensors="pt", max_length=10, truncation=True).to(compute_device)

        if len(inputs['input_ids'][0]) == 10:
            input_ids.append(inputs['input_ids'][0])
            attention_mask.append(inputs['attention_mask'][0])

    inputs = {'input_ids': torch.stack(input_ids), 
                'attention_mask': torch.stack(attention_mask)}
    
    return inputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--wet-file', default=None, type=str)

    args = parser.parse_args()

    compute_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained("gpt2", low_cpu_mem_usage=True).to(compute_device)
    model_s.config.pad_token_id = model_s.config.eos_token_id
    model_s.eval()

    model_m = AutoModelForCausalLM.from_pretrained("gpt2-medium", low_cpu_mem_usage=True).to(compute_device)
    model_m.config.pad_token_id = model_m.config.eos_token_id
    model_m.eval()

    model_xl = AutoModelForCausalLM.from_pretrained("gpt2-xl", low_cpu_mem_usage=True).to(compute_device)
    model_xl.config.pad_token_id = model_xl.config.eos_token_id
    model_xl.eval()

    wet_plain_text = parse_wet_files(args.wet_file)
    
    num_batches = int(math.ceil(args.N / args.batch_size))

    generated_samples = []

    for batch in range(num_batches):
        
        inputs = generate_inputs_internet_text(wet_plain_text, args.batch_size)
        
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
    
    evaluating(generated_samples, model_s, model_m, model_xl, tokenizer, compute_device)