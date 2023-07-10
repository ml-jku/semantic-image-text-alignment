# SITTA: A Semantic Image-Text Alignment for Image Captioning

Fabian Paischer<sup>1 2</sup>,
Thomas Adler<sup>1</sup>,
Markus Hofmarcher<sup>1</sup>,
Sepp Hochreiter<sup>1 2 3</sup>


<sup>1</sup> LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria <br/>
<sup>2</sup> ELLIS Unit Linz  
<sup>3</sup> Institute of Advanced Research in Artificial Intelligence (IARAI)

---

**[SITTA: A semantic Image-Text Alignment for Image Captioning]()** is a lightweight mapping from image to text domain that enables conditioning pretrained Language Models on visual input.
See below some examples for captions created with SITTA for sample images of the MS-COCO validation set.

![Captions](assets/sample_captions.png)

---

## Prerequisites

First clone the repository and create a conda environment with the required packages 

    git clone https://git.bioinf.jku.at/ml/semantic-image-text-alignment.git
    cd semantic-image-text-alignment
    conda env create -f env.yml
    pip install -e .

## Using SITTA for Image Captioning

If you want to use SITTA for image captioning right away, you will first need to dump the token embeddings of the Llama model:

    python semantic_image_text_alignment/data_prep/prepare_embeddings.py --lm-only

Then, you can use SITTA for image captioning within a few lines of code:

    from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer
    import torch
    from semantic_image_text_alignment.pipeline import SITTA
    from PIL import Image
    
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device, index=0)
    sitta = pipeline("sitta-image-to-text", model=model, tokenizer=tokenizer, device=device)
    test_img = Image.open("test_imgs/COCO_val2014_000000334321.jpg")
    sitta(test_sample)

    > 'a white dog sitting on a bench with people sitting around it.'
  
By default SITTA uses the semantic mapping trained via least-squares on MS-COCO data and the RN50x64 CLIP encoder provided [here](https://github.com/openai/CLIP).
All our pre-trained mappings are available in the ```models``` directory.
For now our pipeline only supports the 7 billion huggingface version of Llama.
We will add support for the other language models used in our paper in the future.

## Reproducing Results of our paper

First, download the MS-COCO data and the Flickr30k data and store them in ```data/coco```, and data ```flickr30k```, respectively.

    cd datasets
    mkdir mscoco && cd mscoco
    wget http://images.cocodataset.org/zips/train2014.zip
    unzip train2014.zip
    wget http://images.cocodataset.org/zips/val2014.zip
    unzip val2014.zip
    cd ../..
    
Also, apply for access to the [Flickr30k dataset](https://shannon.cs.illinois.edu/DenotationGraph/) and save the images to ```./datasets/flickr30k```.
Parse both datasets by

    cd data_prep
    python semantic_image_text_alignment/data_prep/parse_coco.py
    python semantic_image_text_alignment/data_prep/parse_flickr30k.py

For computing the different mappings, first, you will need to extract the CLIP and language embeddings for Llama (or other language models) 

    python semantic_image_text_alignment/data_prep/prepare_embeddings.py

You can extract embeddings for the other language models by using the ```--lm``` argument.    
This will run for a while and extract token embeddings for all CLIP backbones and save them to ```data/```.
Next you can train the mappings via *lexical matching* by running

    python semantic_image_text_alignment/train_lexical_matching.py
    
Before running the computation for the *external datasets* method, you will need to run

    python -m spacy download en_core_web_sm
   
This will download and install the english spacy pipeline used for stop-word removal.
Then execute

    python semantic_image_text_alignment/train_external_dataset.py
    
By default the above mappings will be computed for Llama, but you can specify other language models via the ```--lm``` command line argument.
Further you can specify the fraction of the MS-COCO dataset to be used for the computation of the mapping using the ```--fraction``` command line argument.
Currently our code supports ```Llama, T5-v1_1, FLAN-T5, GPT-J, GPT-JT```.
If you want to create mappings for other language models, simply look up the respective huggingface identifier and add it to the code.

To run our retrieval experiments on mscoco, simply run 

    python semantic_image_text_alignment/retrieval_eval.py --mscoco

Finally, you can generate captions for the MS-COCO datasets on the respective test splits via

    python semantic_image_text_alignment/generate_captions.py --k 8 --l 40 --mscoco --vis-encoder RN50x64 --train-method linear_reg --decoding greedy    

For generating captions for the Flickr30k datasets, simply set ```--datadir data/flickr30/imgs_test.pkl``` and ```--flickr```.
The hyperparameters ```k``` and ```l``` denote the number of tokens provided in the prompt, and the number of random permutations, respectively.
Currently, generation supports ```greedy```, ```sampling```, ```nucleus```, and ```topk```.
In case you only have access to small GPUs (VRAM < 48GB) consider using 8-bit quantization by setting ```load_in_8bit=True``` while loading the model from the huggingface hub.

## Pretrained Mappings

We provide the pretrained mappings from all our results in the main paper in the ```models/``` directory.
These include ordinary least squares and procrustes mappings for Llama, GPT-J, GPT-JT, FLAN-T5, and T5-v1_1.


## Results on Retrieval Task

The results for our retrieval task can be found in the ```results/retrieval``` directory.

## Generated Captions

You can find all generated captions, as well as reported scores from our paper for all pretrained mappings and language models on both, the MS-COCO, and Flickr30k datasets, in the ```results/captioning``` directory.
Each result consists of a json file containing the captions for each image in the respective test set, and an associated ```.out``` file containing all computed evaluation metrics.
These metrics (BLEU, CIDEr-D, Rouge-L) are computed using the code from [here](https://github.com/tylin/coco-caption).
The corresponding annotation files for computing these scores can be found in the ```annotations/``` directory.

## LICENSE
MIT LICENSE


