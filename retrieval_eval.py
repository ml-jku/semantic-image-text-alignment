import os
import numpy as np
np.random.seed(101)
from argparse import ArgumentParser
from transformers import LlamaTokenizer
import tqdm
import torch
import pickle
import spacy
from clip.simple_tokenizer import SimpleTokenizer
import clip
from nltk.stem import SnowballStemmer
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils import calc_cosine_sim


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, help='directory where to load the images from', default='data/mscoco/')
    parser.add_argument('--k', type=int, default=10, help='How many tokens to retrieve')
    parser.add_argument('--mscoco', action='store_true', default=False, help='Load mappings trained using MSCoco dataset')
    parser.add_argument('--fraction', default=1, type=float, help='MSCoco fraction to use')
    parser.add_argument('--clip', action='store_true', default=False, help='Evaluate in clip space')
    parser.add_argument('--lm', type=str, default='decapoda-research/llama-7b-hf', help='language model to align to')
    return parser.parse_args()


def compute_dcg_score(src_tokens, tar_tokens):
    tar_tokens = np.array(tar_tokens)
    ranks = np.arange(1, len(src_tokens)+1)
    rels = np.in1d(src_tokens, tar_tokens)
    dcg = np.sum(rels / np.log2(ranks+1))
    n_rels = sum(rels)
    ideal_ranks = np.arange(1, n_rels+1)
    idcg = np.sum(np.ones((n_rels,)) / np.log2(ideal_ranks + 1))
    return dcg / idcg


def main():
    options = create_parser()
    lm = options.lm.split("/")[-1]

    encoders = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 
                'beit_base_patch16_224', 'vit_large_patch16_224_in21k']

    if options.mscoco:
        suffix = f"_{options.fraction}_mscoco"
    else:
        suffix = ''

    if options.clip:
        suffix += '_clip'

    images = pickle.load(open(os.path.join(options.datadir, f'imgs_val.pkl'), 'rb'))
    txts = pickle.load(open(os.path.join(options.datadir, f'txts_val.pkl'), 'rb'))
    keys = list(txts.keys())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sp = spacy.load('en_core_web_sm')
    all_stopwords = sp.Defaults.stop_words
    stemmer = SnowballStemmer(language="english")

    score_dict = {}
    for encoder in encoders:

        if encoder.startswith('beit') or encoder.startswith('vit_large'):
            config = resolve_data_config({}, model=encoder)
            preprocess = create_transform(**config)
            model = timm.create_model(encoder, pretrained=True)
            model.head = torch.nn.Identity()
            model = model.to(device)
            model.eval()
        else:
            model, preprocess = clip.load(encoder)
            model.cuda().eval()

        encoder_clean = encoder.replace("/", "")

        if options.clip:
            target_embs = np.load(f'./data/{encoder_clean}_prompt_embs.npz')
            vocab_indices = np.arange(len(target_embs))
        else:
            all_target_embs = np.load(f'./data/{lm}_embs.npz')
            vocab_indices = np.arange(len(all_target_embs))
            target_embs = (all_target_embs - all_target_embs.mean(0)) / all_target_embs.std(0)

        if options.clip:
            tokenizer = SimpleTokenizer()
        else:
            if 'llama' in lm:
                tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf",
                                                           cache_dir="/system/user/publicdata/llm",
                                                           use_fast=False)
            else:
                raise NotImplementedError(f"{lm} - Language model not supported!!!!")

        stemmed_vocab_toks = np.array([stemmer.stem(tokenizer.decode([t]).strip()) for t in vocab_indices])

        image_features = []
        with torch.no_grad():
            for i in tqdm.trange(0, len(keys), 128):
                ids = keys[i: i + 128]
                if not encoder.startswith('beit') and not encoder.startswith('vit'):
                    batch = torch.stack([preprocess(images[id]) for id in ids]).to(device)
                    embeddings = model.encode_image(batch).float().cpu().numpy()
                else:
                    batch = torch.stack([preprocess(images[id].convert("RGB")) for id in ids]).to(
                        device)
                    embeddings = model(batch).cpu().numpy()
                image_features.append(embeddings)
            image_features = np.concatenate(image_features)

        if not options.clip:
            train_methods = ['linear_reg', 'ridge_reg', 'procrustes', 'robust_procrustes']
        else:
            train_methods = ['clip']

        for train_method in train_methods:

            if not options.clip:
                # load projection matrix and target embs
                if options.mscoco:
                    proj_mat = np.load(os.path.join('./models', f'{lm}_{encoder_clean}_{train_method}_mscoco_{options.fraction}.npy'))
                else:
                    proj_mat = np.load(os.path.join('./models', f'{lm}_{encoder_clean}_{train_method}.npy'))

                image_features = (image_features - image_features.mean(0)) / image_features.std(0)
                proj_features = image_features @ proj_mat
            else:
                proj_features = image_features

            sims = calc_cosine_sim(proj_features, target_embs)
            ranked_sims = np.argsort(sims, axis=-1)[:, ::-1]

            stemmed_caps = []
            ndcgs = []
            for i, key in tqdm.tqdm(enumerate(txts.keys()), desc="Computing score...."):
                caps = ' '.join([w for w in txts[key].split() if w.lower() not in all_stopwords])
                caps = caps.replace('.', '').replace(',', '')
                caps = tokenizer.encode(' '.join(np.unique(caps)))
                if hasattr(tokenizer, 'unk_token_id'):
                    caps = [c for c in caps if c != tokenizer.unk_token_id]
                stemmed_targets = [stemmer.stem(t) for t in tokenizer.decode(caps).split()]
                stemmed_caps.append(stemmed_targets)
                stemmed_rets = stemmed_vocab_toks[ranked_sims[i]]
                ndcg = compute_dcg_score(stemmed_rets, stemmed_targets)
                ndcgs.append(ndcg)

            print('-----------------------------------------------------------------------------------\n')
            print(f"Vision Encoder: {encoder_clean}, Language Model: {lm}, Linear Model: {train_method}\n"),
            print(f"NDCG: {np.mean(ndcgs)}/{np.std(ndcgs)}\n")
            print('-----------------------------------------------------------------------------------\n')

            score_dict[f'{encoder_clean}_{train_method}'] = ndcgs

        rand_ndcgs = []
        for i in tqdm.trange(len(image_features), desc="Computing Random Scores..."):
            perm = np.random.permutation(len(stemmed_vocab_toks))
            stemmed_rets = stemmed_vocab_toks[perm]
            ndcg = compute_dcg_score(stemmed_rets, stemmed_caps[i])
            rand_ndcgs.append(ndcg)

        print(f"Random NDCG: {np.mean(rand_ndcgs)}/{np.std(rand_ndcgs)}")
        print('-----------------------------------------------------------------------------------\n')



if __name__ == '__main__':
    main()

