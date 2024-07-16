# Extracting_training_data_from_GPT2

Performing the experiment as per the [paper](https://arxiv.org/abs/2012.07805)

The objective of the attack is to extract memorized training data from the GPT-2 XL model by leveraging overfitting. Overfitting is when an ML model provides results exceptionally accurate on the training data but poorly on new, unseen data. There are some proven attacks like the membership inference attacks that use this overfitting to successfully extract the training data from the model. The paper improves the available attack by introducing a few more metrics to filter out the unlikely samples, please read the paper for more detailed understanding.

I was able to extract a few samples that were verbatim found on internet. My results: 

## Table of content

- [Methods](#methods)
- [How to use](#how-to-use)
- [Experiment Results](#results-from-my-experiment)

## Methods

In each of the extraction methods the generated samples are ranked according to these six metrics as described in the paper

> - Perplexity: the perplexity of the largest GPT-2 model.
> - Small: the ratio of log-perplexities of the largest GPT-2 model and the Small GPT-2 model.
> - Medium: the ratio as above, but for the Medium GPT-2.
> - zlib: the ratio of the (log) of the GPT-2 perplexity and the zlib entropy (as computed by compressing the text).
> - Lowercase: the ratio of perplexities of the GPT-2 model on the original sample and on the lowercased sample.
> - Window: the minimum perplexity of the largest GPT-2 model across any sliding window of 50 tokens.

When we generate `N` number of samples top 100 in each category are selected and stored. Each sample is stored in a different file with the value of the metric printed at the top.

### Top-k Extraction.

In this approach the Language Model is initialised with only a single token prompt, the token is a special start-of-sentence token and then then use this token to extract 256 tokens with the top-n method with n being 40.

`python3 top_k_extraction.py --N 1000 --batch-size 10`

This will generate 1000 samples and then select top 100 from of the above metrics.

### Prompt from internet text (Common Crawl)

This is the approach where a prompt was given to the model, the prompt was generated from the actual internet text (as GPT-2’s training data is internet text). This internet text that it uses to generate prompt was from the common crawl data (CC). 

Please visit [Common Crawl](https://commoncrawl.org/overview) to download your preferred `.wet` file. I used CC files from 2015-2017 as (as this was when GPT-2 was being trained).

`python3 internet_text_extraction.py --N 1000 --batch_size 10 --wet_file cc_2017.warc.wet`

This will generate 1000 samples with by randomly selecting one entry from the `cc_2017.warc.wet` file for each sample and generating first 20 tokens from that data. Then select top 100 from of the above metrics.

### Sampling with decaying temperature

We notice that in the previous approach we find a lot of extracted data with a very high k in k-eidetic memorization, like extracting numeric values of certain sequence and dates in a calender. So we try artificially flattening the probability distribution by making it less confident by changing the output softmax to softmax(z/t) where temperature(t) > 1, the model will be less confident for higher t values so we apply a decaying temperature this way the model will not go too off. I started the with t value from 10 and brought to to 1.

`python3 temperature_decay_extraction.py --N 1000 --batch-size 10`

This will generate 1000 samples and then select top 100 from of the above metrics.

It takes `--wet_file` as an options argument.

## How to use

Clone the repository and then, install all the requirements.

`pip3 install -r requirements.txt`

- To use top_k_extraction create a directory `top_k_output` in the directory top_k_extraction.py is present.
    - `python3 top_k_extraction.py --N <num-of-samples> --batch-size <batch-size>`

- To use internet_text_extraction create a directory `internet_text_output` in the directory internet_text_extraction.py is present.
    - `python3 internet_text_extraction.py --N <num-of-samples> --batch_size <batch-size> --wet_file <cc-file>`

- To use temperature_decay_extraction create a directory `temp_output` in the directory temperature_decay_extraction.py is present.
    - `python3 temperature_decay_extraction.py --N <num-of-samples> --batch_size <batch-size>`

## Results from my experiment

#### Attack overview

With the above attacks in mind I generated 1000 samples each of length 256 tokens with each of the three text generation methods above. So each of the above method generates 600 samples (100 for each metric). So in total `3 * 600 = 1800` samples were reviewed manually.

See the below table describing all the results from different methods.

#### The memorised content description

The generated memorised content was also very different in the different methods. The memorised content found in the top-n method was usually of high k value in k-eidetic memorization, but only one example where the k value was smaller. The internet text as a prompt method on the other had created a lot more examples with smaller k and also this method memorised a few URLs verbatim that are correctly resolving.

Please see the below image showing change in perplexity values in different method

#### The metric values in different method

Not only did the internet text as a prompt method generate more memorised content, it also created more samples with good metrics. That is the value of perplexity for most of the generated samples was low compared to the top-n method.

Please see the below images which compares these metrics.


#### Some other things to note

I found that with the increase of the prompt length the number of memorised contents was also increased, but the value was very small so I don’t know if we can generalise this.

I also found the increasing the sample length with more number of tokens did not give longer memorised contents, most of the memorization was in the staring of the sample.

## I will soon include interesting examples of the memorised samples