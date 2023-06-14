import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from argparse import ArgumentParser
from transformers import AutoTokenizer
import numpy as np
from utils import orthogonal_procrustes
import tqdm
import os
import pickle
from sklearn.linear_model import LinearRegression, Ridge
import clip
import spacy
from transformers import LlamaTokenizer
import torch
import time


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--lm', type=str, default='decapoda-research/llama-7b-hf')
    parser.add_argument('--fraction', type=float, default=1, help="Fraction of data to use for training of the mapping")
    parser.add_argument('--dataset', type=str, required=True, choices=["mscoco", "flickr30k"], default=1, help="Fraction of data to use for training of the mapping")
    return parser.parse_args()


def tokenize(txts, keys, tokenizer, stopwords, unk_token_id=None):
    tokenized = []
    for key in tqdm.tqdm(keys):
        caps = ' '.join([w for w in txts[key].split() if w.lower() not in stopwords])
        caps = caps.replace('.', '').replace(',', '').split()
        caps = tokenizer.encode(' '.join(np.unique(caps)))
        caps = [c for c in caps if c != unk_token_id]
        tokenized.append(caps)
    return tokenized


def main():
    sp = spacy.load('en_core_web_sm')
    all_stopwords = sp.Defaults.stop_words
    options = create_parser()
    assert options.fraction <= 1 and options.fraction > 0, "Fraction of training data must be within (0,1]"

    txts_train = pickle.load(open(os.path.join('./data', options.dataset, f'txts_train.pkl'), 'rb'))
    if options.fraction == 1:
        subset_keys = list(txts_train.keys())
    else:
        np.random.seed(101)
        subset_keys = np.random.choice(list(txts_train.keys()), size=int(len(txts_train) * options.fraction))

    print(f"Using subset of {len(subset_keys)} images")

    if 'llama' in options.lm:
        tokenizer = LlamaTokenizer.from_pretrained(options.lm, cache_dir="/system/user/publicdata/llm", use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(options.lm, cache_dir="/system/user/publicdata/llm", use_fast=False)
    unk_id = tokenizer.unk_token_id
    tokenized_train = tokenize(txts_train, subset_keys, tokenizer, all_stopwords, unk_id)
    lm = options.lm.split("/")[-1]

    all_target_embs = np.load(f'./data/{lm}_embs.npz')
    target_mean = all_target_embs.mean(0)
    target_std = all_target_embs.std(0)
    targets = (all_target_embs - target_mean) / target_std

    encoders = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14',
                    'ViT-L/14@336px', 'beit_base_patch16_224', 'vit_large_patch16_224_in21k']

    for encoder in encoders:

        encoder_clean = encoder.replace('/', '')
        print(f"Model: {encoder}")
        if not os.path.exists(f'./data/{encoder_clean}_{lm}_{options.dataset}_{options.fraction}_embs.npy'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Constructing {options.dataset} dataset...")

            if encoder.startswith('beit') or encoder.startswith('vit'):
                config = resolve_data_config({}, model=encoder_clean)
                preprocess = create_transform(**config)
                model = timm.create_model(encoder_clean, pretrained=True)
                model.head = torch.nn.Identity()
                model = model.to(device)
            else:
                model, preprocess = clip.load(encoder)
                model.cuda().eval()

            images_train = pickle.load(open(os.path.join('./data', options.dataset, f'imgs_train.pkl'), 'rb'))
            image_features = []
            with torch.no_grad():
                for i in tqdm.trange(0, len(subset_keys), 128):
                    ids = subset_keys[i: i + 128]
                    if encoder.startswith('beit') or encoder.startswith('vit'):
                        batch = torch.stack([preprocess(images_train[id].convert("RGB")) for id in ids]).to(device)
                        embeddings = model(batch).cpu().numpy()
                    else:
                        batch = torch.stack([preprocess(images_train[id]) for id in ids]).to(device)
                        embeddings = model.encode_image(batch).cpu().numpy()
                    image_features.append(embeddings)
                image_features = np.concatenate(image_features)

            src_embs = []
            tar_embs = []
            train_cls = []
            for i in tqdm.trange(len(image_features), desc="Loading correspondences..."):
                labels = targets[tokenized_train[i]]
                tar_embs.extend(labels)
                src_embs.append(image_features[i])
                train_cls.append(np.repeat(i, len(labels), axis=0))

            # dump embeddings
            src_embs = np.array(src_embs, dtype=np.float32).squeeze()
            tar_embs = np.array(tar_embs)
            train_cls = np.concatenate(train_cls)
            np.save(f'./data/{encoder_clean}_{lm}_{options.dataset}_{options.fraction}_embs', src_embs)
            np.save(f'./data/{encoder_clean}_{lm}_{options.dataset}_{options.fraction}_targets', tar_embs)
            np.save(f'./data/{encoder_clean}_{lm}_{options.dataset}_{options.fraction}_train_cls', train_cls)
        else:
            src_embs = np.load(f'./data/{encoder_clean}_{lm}_{options.dataset}_{options.fraction}_embs.npy')
            tar_embs = np.load(f'./data/{encoder_clean}_{lm}_{options.dataset}_{options.fraction}_targets.npy')
            train_cls = np.load(f'./data/{encoder_clean}_{lm}_{options.dataset}_{options.fraction}_train_cls.npy')

        # tar_embs are already centered and scaled
        src_embs = (src_embs - src_embs.mean(0)) / src_embs.std(0)
        src_embs = src_embs[train_cls]

        for train_method in ['linear_reg', 'ridge_reg', 'procrustes', 'robust_procrustes']:

            if not os.path.exists(os.path.join('./models', f'{lm}_{encoder_clean}_{train_method}_{options.dataset}_{options.fraction}.npy')):

                # fit linear model
                if train_method == 'procrustes':
                    start = time.time()
                    proj_mat = orthogonal_procrustes(src_embs, tar_embs)
                    end = time.time()
                elif train_method == 'robust_procrustes':
                    # Robust procrustes method from https://arxiv.org/abs/2205.11616
                    start = time.time()
                    eps = 1e-3
                    m = 5
                    proj_mat = orthogonal_procrustes(src_embs, tar_embs)
                    for j in range(m):
                        weights = 1 / (np.linalg.norm(tar_embs - (src_embs @ proj_mat), ord=2, axis=-1) + eps)
                        weights = (weights / np.max(weights))**.5
                        weights = weights.reshape(-1, 1)
                        proj_mat = orthogonal_procrustes(weights * src_embs, weights * tar_embs)
                    end = time.time()
                elif train_method == 'linear_reg':
                    start = time.time()
                    model = LinearRegression(fit_intercept=False)
                    model.fit(src_embs, tar_embs)
                    end = time.time()
                    proj_mat = model.coef_.T
                elif train_method == 'ridge_reg':
                    start = time.time()
                    model = Ridge(alpha=1., fit_intercept=False)
                    model.fit(src_embs, tar_embs)
                    end = time.time()
                    proj_mat = model.coef_.T
                else:
                    raise NotImplementedError(f'{train_method} - Training method not supported!!')

                print(f"Elapsed time: {end - start}")
                np.save(os.path.join('./models', f'{lm}_{encoder_clean}_{train_method}_{options.dataset}_{options.fraction}'), proj_mat)


if __name__ == '__main__':
    main()
