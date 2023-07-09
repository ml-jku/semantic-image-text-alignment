import os
import numpy as np
np.random.seed(101)
from argparse import ArgumentParser
from transformers import AutoTokenizer, GPTJForCausalLM,\
    AutoModelForCausalLM, AutoModelWithLMHead, T5ForConditionalGeneration, LlamaForCausalLM, LlamaTokenizer
import tqdm
import torch
import pickle
import clip
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch.nn.functional as nnf
from torch.distributions import Categorical
import json


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/mscoco/imgs_val.pkl', help='path to images')
    parser.add_argument('--k', type=int, default=5, help='How many tokens to retrieve')
    parser.add_argument('--l', type=int, default=5, help='How many captions to sample')
    parser.add_argument('--mscoco', action='store_true', default=False, help='Load mappings trained using MSCoco dataset')
    parser.add_argument('--flickr', action='store_true', default=False, help='Load mappings trained using Flickr30k dataset')
    parser.add_argument('--fraction', default=1, type=float, help='MSCoco fraction to use')
    parser.add_argument('--lm', type=str, default='decapoda-research/llama-7b-hf', help='language model to align to')
    parser.add_argument('--vis-encoder', type=str, default='ViT-B/16',
                        choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14',
                                 'ViT-L/14@336px', 'beit_base_patch16_224', 'vit_large_patch16_224_in21k'],
                        help='Vision encoder to use')
    parser.add_argument('--train-method', help='Method that was used to train the semantic mapping')
    parser.add_argument('--decoding', type=str, choices=['greedy', 'sampling', 'topk', 'nucleus'], required=True,
                        help='What decoding strategy to use')
    return parser.parse_args()


def calc_cos_sims(queries, features):
    normed_queries = queries / np.linalg.norm(queries, ord=2, axis=-1, keepdims=True)
    normed_features = features / np.linalg.norm(features, ord=2, axis=-1, keepdims=True)
    return normed_queries @ normed_features.T


def generate(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    decoder_embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.95,
    top_k=640,
    decoding='greedy',
    temperature=1.0,
    stop_tokens: str = [".", "\n"],
):
    model.eval()
    generated_list = []
    if isinstance(model, T5ForConditionalGeneration):
        stop_token_index = [1]
    elif isinstance(model, LlamaForCausalLM):
        stop_token_index = [tokenizer.encode(stop_token)[-1] for stop_token in stop_tokens]
        stop_token_index = stop_token_index + [29889, 13]
    else:
        stop_token_index = [tokenizer.encode(stop_token)[0] for stop_token in stop_tokens]

    filter_value = -float("Inf")
    device = next(model.parameters()).device
    bs, seqlen, *_ = embed.shape
    terminated = np.full(shape=(bs,), fill_value=False, dtype=np.bool8)

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if decoder_embed is not None:
                generated = decoder_embed
            elif embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                if isinstance(model, T5ForConditionalGeneration):
                    generated = model.shared(tokens)
                elif isinstance(model, LlamaForCausalLM):
                    generated = model.model.embed_tokens(tokens)
                else:
                    generated = model.transformer.word_emb(tokens)


            for i in range(entry_length):

                if not isinstance(model, T5ForConditionalGeneration):
                    outputs = model(inputs_embeds=generated)
                else:
                    outputs = model(inputs_embeds=embed, decoder_inputs_embeds=generated)

                logits = outputs.logits[:, -1, :]

                if decoding == 'greedy':
                    # greedy decoding
                    next_token = torch.argmax(logits, -1).unsqueeze(0)
                elif decoding == 'nucleus':
                    # nucleus sampling
                    logits = logits / (temperature if temperature > 0 else 1.0)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = [sorted_indices[i, si] for i, si in enumerate(sorted_indices_to_remove)]
                    for i, ind in enumerate(indices_to_remove):
                        logits[i][ind] = filter_value

                    dist = Categorical(logits=logits)
                    next_token = dist.sample().unsqueeze(0)
                elif decoding == 'sampling':
                    # primitive sampling
                    logits = logits / (temperature if temperature > 0 else 1.0)
                    dist = Categorical(logits=logits)
                    next_token = dist.sample().unsqueeze(0)
                else:
                    # topk sampling
                    logits = logits / (temperature if temperature > 0 else 1.0)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_indices = sorted_indices[:, :top_k]
                    logits = logits.gather(-1, sorted_indices)
                    dist = Categorical(logits=logits)
                    next_token = sorted_indices.gather(-1, dist.sample().unsqueeze(0))

                terminated[np.in1d(next_token.cpu().numpy(), stop_token_index)] = True

                if isinstance(model, T5ForConditionalGeneration):
                    next_token_embed = model.shared(next_token)
                elif isinstance(model, LlamaForCausalLM):
                    next_token_embed = model.model.embed_tokens(next_token)
                else:
                    next_token_embed = model.transformer.wte(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=0)
                generated = torch.cat((generated, next_token_embed.transpose(0, 1)), dim=1)
                # if next_token.item() in stop_token_index:
                #     break

                if terminated.any():
                    dones = np.nonzero(terminated)[0]
                    for done in dones:
                        output_text = tokenizer.decode(tokens[:, done])
                        generated_list.append(output_text)

                    generated = generated[~terminated]
                    tokens = tokens[:, ~terminated]
                    if not len(generated):
                        # All beams are terminated
                        break
                    terminated = np.full(shape=(generated.shape[0],), fill_value=False, dtype=np.bool8)

        if len(generated) > 0:
            # append leftover captions to generated list
            dones = np.nonzero(~terminated)[0]
            for done in dones:
                output_text = tokenizer.decode(tokens[:, done])
                generated_list.append(output_text)

    return generated_list


def main():
    options = create_parser()
    if "flan" in options.lm or "GPT-JT" in options.lm:
        prompt = "Generate a caption containing the following words: "
    else:
        prompt = "A picture of"

    encoder = options.vis_encoder
    lm = options.lm.split('/')[-1]

    if options.mscoco:
        suffix = f"{options.fraction}_mscoco"
    elif options.flickr:
        suffix = f"{options.fraction}_flickr"
    else:
        suffix = ''

    suffix += f'_ntoks_{options.k}_nsamples_{options.l}'

    if options.train_method:
        suffix += f'_{options.train_method}'

    dataset = "flickr30k" if "flickr" in options.datadir else "mscoco"
    suffix += f'_{options.decoding}_{dataset}'
    if 'val' in options.datadir:
        set = '_val'
    elif 'test' in options.datadir:
        set = '_test'
    else:
        set = '_train'

    suffix += set

    if options.datadir.endswith('pkl'):
        images = pickle.load(open(options.datadir, 'rb'))
    elif options.datadir.endswith('.npy'):
        images = {str(i): Image.fromarray(f) for i, f in enumerate(np.load(options.datadir))}
    else:
        raise NotImplementedError(f"Not able to load from {options.datadir}")

    keys = list(images.keys())
    os.makedirs('results', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if encoder.startswith('beit') or encoder.startswith('vit_large'):
        config = resolve_data_config({}, model=encoder)
        preprocess = create_transform(**config)
        model = timm.create_model(encoder, pretrained=True)
        model.head = torch.nn.Identity()
        model = model.to(device)
        model.eval()

        selection_model, sel_preprocess = clip.load("RN50x64")
        selection_model = selection_model.to(device)
        selection_model.eval()
    else:
        model, preprocess = clip.load(encoder)
        model = model.to(device)
        model.eval()

    encoder_clean = encoder.replace("/", "")
    transformer_embs = np.load(f'./data/{lm}_embs.npz')
    tr_mean = transformer_embs.mean(0)
    tr_std = transformer_embs.std(0)
    transformer_embs = (transformer_embs - tr_mean) / tr_std

    if 'gpt-j' in lm:
        transformer = GPTJForCausalLM.from_pretrained(
            options.lm, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            cache_dir="/system/user/publicdata/llm"
        )
        transformer = transformer.to(device)
        tokenizer = AutoTokenizer.from_pretrained(options.lm,
                                                  cache_dir="/system/user/publicdata/llm")
    elif 'GPT-JT' in lm:
        transformer = AutoModelForCausalLM.from_pretrained(
            "togethercomputer/GPT-JT-6B-v1", torch_dtype=torch.float32, low_cpu_mem_usage=True,
            cache_dir="/system/user/publicdata/llm"
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1",
                                                  cache_dir="/system/user/publicdata/llm")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif 'flan-t5' in lm:
        if 'xxl' in lm:
            # load 8-bit quantized model
            transformer = AutoModelWithLMHead.from_pretrained(f"google/{options.lm}", device_map='auto',
                                                              offload_folder="offload", load_in_8bit=True,
                                                              cache_dir="/system/user/publicdata/llm")
        else:
            transformer = AutoModelWithLMHead.from_pretrained(
                f"google/{options.lm}", torch_dtype=torch.float32, low_cpu_mem_usage=True,
                cache_dir="/system/user/publicdata/llm"
            )
            transformer = transformer.to(device)
        tokenizer = AutoTokenizer.from_pretrained(f"google/{options.lm}", cache_dir="/system/user/publicdata/llm")
    elif 't5-v1_1' in lm:
        if 'xxl' in lm:
            # load 8-bit quantized model
            transformer = AutoModelWithLMHead.from_pretrained(f"google/{options.lm}", device_map='auto',
                                                              offload_folder="offload", load_in_8bit=True,
                                                              cache_dir="/system/user/publicdata/llm")
        else:
            transformer = AutoModelWithLMHead.from_pretrained(
                f"google/{options.lm}", torch_dtype=torch.float32, low_cpu_mem_usage=True,
                cache_dir="/system/user/publicdata/llm"
            )
            transformer = transformer.to(device)
        tokenizer = AutoTokenizer.from_pretrained(f"google/{options.lm}", cache_dir="/system/user/publicdata/llm",
                                                  use_fast=False)
    elif 'llama' in lm:
        transformer = LlamaForCausalLM.from_pretrained(
            "decapoda-research/llama-7b-hf", torch_dtype=torch.float32, low_cpu_mem_usage=True,
            cache_dir="/system/user/publicdata/llm"
        )
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf",
                                                   cache_dir="/system/user/publicdata/llm",
                                                   use_fast=False)
        transformer = transformer.to(device)
    else:
        raise NotImplementedError(f"{lm} - Language model not supported!!!!")

    transformer.eval()

    if not isinstance(transformer, T5ForConditionalGeneration):
        encoded_prompt = tokenizer.encode(prompt)

        with torch.no_grad():

            if isinstance(transformer, LlamaForCausalLM):
                prompt_embs = transformer.model.embed_tokens(torch.LongTensor(encoded_prompt).to(device)).to(device)
            else:
                prompt_embs = transformer.transformer.wte(torch.LongTensor(encoded_prompt).to(device))

    if not os.path.exists(f'./data/{dataset}/image_features_{encoder_clean}{set}.npy'):
        image_features = []
        with torch.no_grad():
            for i in tqdm.trange(0, len(keys), 16):
                ids = keys[i: i + 16]
                if not encoder.startswith('beit') and not encoder.startswith('vit'):
                    batch = torch.stack([preprocess(images[id]) for id in ids]).to(device)
                    embeddings = model.encode_image(batch).float().cpu().numpy()
                else:
                    batch = torch.stack([preprocess(images[id].convert("RGB")) for id in ids]).to(
                        device)
                    embeddings = model(batch).cpu().numpy()
                image_features.append(embeddings)
            image_features = np.concatenate(image_features)
            np.save(f'./data/{dataset}/image_features_{encoder_clean}{set}', image_features)
    else:
        image_features = np.load(f'./data/{dataset}/image_features_{encoder_clean}{set}.npy')

    if encoder.startswith('vit') or encoder.startswith('beit'):
        selection_img_features = np.load(f'./data/{dataset}/image_features_RN50x64{set}.npy')

    orig_image_features = image_features.copy()
    image_features = (image_features - image_features.mean(0)) / image_features.std(0)

    if options.mscoco:
        proj_mat = np.load(os.path.join('models', f'{lm}_{encoder_clean}_{options.train_method}_mscoco_{options.fraction}.npy'))
    elif options.flickr:
        proj_mat = np.load(os.path.join('models', f'{lm}_{encoder_clean}_{options.train_method}_flickr30k_{options.fraction}.npy'))
    else:
        proj_mat = np.load(os.path.join('models', f'{lm}_{encoder_clean}_{options.train_method}.npy'))

    proj_features = image_features @ proj_mat

    sims = calc_cos_sims(proj_features, transformer_embs)
    ranked_sims = np.argsort(sims, axis=-1)[:, ::-1]
    top_toks = []
    for i in range(len(ranked_sims)):
        dec = [tokenizer.decode([tok]).strip().lower() for tok in ranked_sims[i][:50]]
        unique, idx = np.unique(dec, return_index=True)
        top_toks.append(ranked_sims[i][:50][np.sort(idx)][:options.k])

    ann_file = []
    for i, key in tqdm.tqdm(enumerate(keys), desc="Creating Captions...."):

        # cur_img_emb = proj_features[i]
        cur_img_emb = top_toks[i].copy()
        if 'gpt-j' in lm or 'llama' in lm:
            insert_token = tokenizer.encode(' ')
            if len(insert_token) > 1:
                # avoid beginning of sequence token
                del insert_token[0]
            cur_img_emb = np.insert(cur_img_emb, np.arange(1, options.k+1), insert_token*(options.k))

        if not isinstance(transformer, T5ForConditionalGeneration) and not "GPT-JT" in options.lm:
            permuted_prompts = [cur_img_emb]
        else:
            permuted_prompts = [prompt + ' '.join([tokenizer.decode(tok) for tok in cur_img_emb])]
        # generate permutations
        for _ in np.arange(options.l - 1):
            if 'gpt-j' in lm or 'llama' in lm:
                perm = np.random.permutation(np.arange(0, options.k*2, 2))
                perm = np.insert(perm, np.arange(1, options.k + 1), np.arange(1, options.k * 2, 2))
            else:
                perm = np.random.permutation(np.arange(options.k))
            if not isinstance(transformer, T5ForConditionalGeneration) and not "GPT-JT" in options.lm:
                permuted_prompts.append(cur_img_emb[perm])
            else:
                if 't5-v1_1' in lm:
                    permuted_prompts.append(' '.join([tokenizer.decode(tok) for tok in cur_img_emb[perm]]) + prompt)
                else:
                    permuted_prompts.append(prompt + ' '.join([tokenizer.decode(tok) for tok in cur_img_emb[perm]]))

        if not isinstance(transformer, T5ForConditionalGeneration) and not "GPT-JT" in options.lm:
            permuted_prompts = np.array(permuted_prompts)

        with torch.no_grad():
            if isinstance(transformer, LlamaForCausalLM):
                prompt_prefix = transformer.model.embed_tokens(torch.LongTensor(permuted_prompts).to(device)).to(device)
            elif hasattr(transformer, 'transformer') and not "GPT-JT" in options.lm:
                prompt_prefix = transformer.transformer.wte(torch.LongTensor(permuted_prompts).to(device))

            if not isinstance(transformer, T5ForConditionalGeneration) and not "GPT-JT" in options.lm:
                cur_prompt = torch.cat([prompt_prefix, prompt_embs.unsqueeze(0).expand(options.l, *prompt_embs.shape)], dim=1)
            else:
                encoding = tokenizer(permuted_prompts, padding=True, return_tensors='pt')
                for k in encoding.keys():
                    encoding[k] = encoding[k].to(device)

        # prompt the LM to generate text
        clip_sc = []
        with torch.no_grad():
            if not isinstance(transformer, T5ForConditionalGeneration) and not "GPT-JT" in options.lm:
                caps = generate(transformer, tokenizer, embed=cur_prompt, decoding=options.decoding)
            else:
                generated = transformer.generate(**encoding, max_length=67)
                caps = tokenizer.batch_decode(generated, skip_special_tokens=True)

            caps = np.unique(caps)
            embeds = []
            for cap in caps:
                if not encoder.startswith('vit') and not encoder.startswith('beit'):
                    embeds.append(model.encode_text(clip.tokenize(cap, truncate=True).to(device)).cpu().numpy().squeeze())
                else:
                    embeds.append(selection_model.encode_text(clip.tokenize(cap, truncate=True).to(device)).cpu().numpy().squeeze())

            embeds = np.array(embeds)
            if not encoder.startswith('vit') and not encoder.startswith('beit'):
                sims = calc_cos_sims(orig_image_features[i].reshape(1, -1), embeds)
            else:
                sims = calc_cos_sims(selection_img_features[i].reshape(1, -1), embeds)

            generated_cap = caps[np.argmax(sims)]
            clip_sc.append(np.max(sims)*2.5)

        image_id = key.split("/")[-1].split('.')[0].split('_')[-1].lstrip('0')
        ann_file.append({'image_id': image_id, 'caption': generated_cap})

    print(f"CLIP Score: {np.mean(clip_sc)}/{np.std(clip_sc)}")
    if options.avalon:
        env = options.datadir.split('/')[-1].split('.')[0].split('_')[-1]
        json.dump(ann_file, open(f'results/captions_{env}_{encoder_clean}_{lm}_{suffix}.json', 'w'))
    else:
        json.dump(ann_file, open(f'results/captions_val_{encoder_clean}_{lm}_{suffix}.json', 'w'))


if __name__ == '__main__':
    main()

