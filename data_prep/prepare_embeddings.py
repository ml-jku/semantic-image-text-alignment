from transformers import \
    TransfoXLTokenizer, \
    TransfoXLModel, \
    GPTJModel, \
    AutoTokenizer, \
    PreTrainedTokenizerFast, \
    AutoModelForCausalLM, \
    AutoModelWithLMHead, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from argparse import ArgumentParser
import numpy as np
import torch
import os
import tqdm
import sys
sys.path.append('..')
from variables import imagenet_templates


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--lm', default='llama-7b-hf', help='Language Model to extract embeddings for')
    return parser.parse_args()


def get_vocab(tokenizer, vocab_size):
    vocab = []
    if isinstance(tokenizer, SimpleTokenizer):
        for i in range(vocab_size):
            vocab.append(tokenizer.decode([i]))
        vocab = np.array(vocab)
    else:
        vocab = np.array(tokenizer.convert_ids_to_tokens(np.arange(vocab_size)))
    # else:
    #    raise NotImplementedError(f"{tokenizer} not known!!!")
    return vocab


def get_clip_embs(tokenized, clip_net, device='cuda', batch_size=128):
    clip_embs = []
    for i in range(0, len(tokenized), batch_size):
        with torch.no_grad():
            tok_emb = clip_net.encode_text(tokenized[i:i+batch_size].to(device))
            clip_embs.append(tok_emb.float().cpu().numpy())
    clip_embs = np.concatenate(clip_embs)
    return clip_embs


def main():
    if not os.path.exists('./data'):
        os.makedirs('./data', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    options = create_parser()
    lm = options.lm

    if 'gpt-j' in lm:
        transformer = GPTJModel.from_pretrained(
            "EleutherAI/gpt-j-6B", torch_dtype=torch.float32, low_cpu_mem_usage=True,
            cache_dir="/system/user/publicdata/llm"
        )
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="/system/user/publicdata/llm")
        n_tokens = tokenizer.vocab_size
    elif 'GPT-JT' in lm:
        transformer = AutoModelForCausalLM.from_pretrained(
            "togethercomputer/GPT-JT-6B-v1", torch_dtype=torch.float32, low_cpu_mem_usage=True,
            cache_dir="/system/user/publicdata/llm"
        )
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1", cache_dir="/system/user/publicdata/llm")
        n_tokens = tokenizer.vocab_size
    elif 'flan-t5-base' in lm:
        transformer = AutoModelWithLMHead.from_pretrained(
            "google/flan-t5-base", torch_dtype=torch.float32, low_cpu_mem_usage=True,
            cache_dir="/system/user/publicdata/llm"
        )
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir="/system/user/publicdata/llm")
        n_tokens = tokenizer.vocab_size
    elif 't5-v1_1' in lm:
        transformer = AutoModelWithLMHead.from_pretrained(
            "google/t5-v1_1-base", torch_dtype=torch.float32, low_cpu_mem_usage=True,
            cache_dir="/system/user/publicdata/llm"
        )
        tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base", cache_dir="/system/user/publicdata/llm",
                                                  use_fast=False)
        n_tokens = tokenizer.vocab_size
    elif 'llama' in lm:
        transformer = LlamaForCausalLM.from_pretrained(
            "decapoda-research/llama-7b-hf", torch_dtype=torch.float32, low_cpu_mem_usage=True,
            cache_dir="/system/user/publicdata/llm"
        )
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", cache_dir="/system/user/publicdata/llm",
                                                  use_fast=False)
        n_tokens = tokenizer.vocab_size
    else:
        raise NotImplementedError(f"{lm} - Language model not supported!!!!")

    if not os.path.exists(f'../data/{lm}_embs.npz'):
        print("Dumping LM vocab...")
        with torch.no_grad():
            if lm == 'transfo-xl-wt103':
                all_embs = transformer.word_emb(torch.arange(n_tokens)).cpu().numpy()
            elif 'llama' in lm or 'alpaca' in lm:
                all_embs = transformer.model.embed_tokens(torch.arange(n_tokens).unsqueeze(-1)).cpu().numpy()
            elif 'gpt' in lm:
                all_embs = transformer.wte(torch.arange(n_tokens).unsqueeze(-1)).cpu().numpy()
            elif 't5' in lm:
                all_embs = transformer.shared(torch.arange(n_tokens).unsqueeze(-1)).cpu().numpy()
            else:
                all_embs = transformer.transformer.wte(torch.arange(n_tokens).unsqueeze(-1)).cpu().numpy()

            # dump transformer embeddings
            np.save(open(f'../data/{lm}_embs.npz', 'wb'), all_embs.squeeze())

    clip_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

    for encoder in clip_models:

        model, preprocess = clip.load(encoder)
        encoder = encoder.replace("/", "")

        model.cuda().eval()
        clip_vocab_size = model.vocab_size
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        print("Vocab size:", clip_vocab_size)
        clip_tokenizer = SimpleTokenizer()

        if not os.path.exists('../data/clip_vocab.npy'):
            print("Dumping vocab...")
            clip_vocab = get_vocab(clip_tokenizer, clip_vocab_size)
            np.save('../data/clip_vocab', clip_vocab)
        else:
            clip_vocab = np.load('../data/clip_vocab.npy')

        if not os.path.exists(f'../data/{encoder}_{lm}_prompt_embs.npz'):
            print("Dumping prompted LM tokens in CLIP space...")
            clip_embs = []
            transformer_vocab = get_vocab(tokenizer, tokenizer.vocab_size)
            for tok in tqdm.tqdm(transformer_vocab):
                prompted = [p.format(tok) for p in imagenet_templates]
                tokenized = clip.tokenize(prompted, truncate=True).to(device)
                with torch.no_grad():
                    vecs = model.encode_text(tokenized).cpu().mean(0).numpy()
                clip_embs.append(vecs)
            clip_embs = np.array(clip_embs)
            np.save(open(f'../data/{encoder}_{lm}_prompt_embs.npz', 'wb'), clip_embs.squeeze())

        if not os.path.exists(f'../data/{encoder}_prompt_embs.npz'):
            print("Dumping prompted embeddings...")
            clip_embs = []
            for tok in tqdm.tqdm(clip_vocab):
                prompted = [p.format(tok) for p in imagenet_templates]
                tokenized = clip.tokenize(prompted).to(device)
                with torch.no_grad():
                    vecs = model.encode_text(tokenized).cpu().mean(0).numpy()
                clip_embs.append(vecs)
            clip_embs = np.array(clip_embs)
            np.save(open(f'../data/{encoder}_prompt_embs.npz', 'wb'), clip_embs.squeeze())


if __name__ == '__main__':
    main()