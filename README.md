# Semantic Image Text Alignment

To reproduce our results, first download the annotations for MS-COCO and Flickr30k from [here](https://cs.stanford.edu/people/karpathy/deepimagesent/) and unzip them to ```./annotations```.
Download the MS-COCO [training data](http://images.cocodataset.org/zips/train2014.zip), MS-COCO [validation data](http://images.cocodataset.org/zips/val2014.zip) and unpack the raw images to ```./datasets/mscoco```.
Also, apply for access to the [Flickr30k dataset](https://shannon.cs.illinois.edu/DenotationGraph/) and save the images to ```./datasets/flickr30k```.

Then create a conda environment with the required packages 

    conda env create -f env.yml

After installing the conda environment parse both datasets by

    cd data_prep
    python parse_coco.py
    python parse_flickr30k.py

For computing the mapping via *lexical matching* first, extract the CLIP and language embeddings for Llama 

    python get_embeddings.py

then you can train the mappings via

    cd ..
    python train_lexical_matching.py
    
For computing the mapping via *external datasets* execute

    python train_external_dataset.py
    
Finally, you can generate captions for the MS-COCO datasets on the respective test splits via

    python generate_captions.py --k 8 --l 40 --mscoco --vis-encoder RN50x64 --train-method linear_reg --decoding greedy    

For generating captinos for the Flickr30k datasets, simply set ```--datadir data/flickr30/imgs_test.pkl``` and ```--flickr```.
The hyperparameters ```k``` and ```l``` denote the number of tokens provided in the prompt, and the number of random permutations, respectively.
Currently, generation supports ```greedy```, ```sampling```, ```nucleus```, and ```topk```.
In case you only have access to small GPUs please consider using 8-bit quantization by setting ```load_in_8bit=True``` while loading the model from the huggingface hub.
