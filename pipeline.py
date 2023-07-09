from transformers import Pipeline, CLIPTokenizer, LlamaForCausalLM, LlamaTokenizer
import clip
from clip.simple_tokenizer import SimpleTokenizer
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import pipeline
import numpy as np
import torch
from PIL import Image
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Path to image")
    return parser.parse_args()


class SITTA(Pipeline):

    def __init__(self, model, tokenizer, device, framework=None, task='sitta-image-to-text'):
        super(SITTA, self).__init__(model=model, tokenizer=tokenizer, task=task, framework=framework, device=device)
        self.vis_encoder, self.custom_processor = clip.load("RN50x64", download_root="/system/user/publicdata/llm")
        self.vis_encoder.eval()
        self.vis_encoder = self.vis_encoder.to(self.device)
        self.clip_tokenizer = SimpleTokenizer()
        self.projection = np.load("models/llama-7b-hf_RN50x64_linear_reg_mscoco_1.0.npy")
        self.targets = np.load("data/llama-7b-hf_embs.npz")
        # Hardcode stds for MS-COCO datasets
        self.coco_mean, self.coco_std = np.load("data/coco_stats.npy")

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        if "topk" in kwargs:
            preprocess_kwargs["topk"] = kwargs["topk"]
        if "m" in kwargs:
            preprocess_kwargs["m"] = kwargs["m"]
            forward_kwargs["m"] = kwargs["m"]
        if "length" in kwargs:
            forward_kwargs["length"] = kwargs["length"]
            forward_kwargs["m"] = kwargs["m"]
        return preprocess_kwargs, forward_kwargs, {}

    def _calc_cos_sims(self, source_embs, target_embs):
        normed_source_embs = source_embs / np.linalg.norm(source_embs, ord=2, axis=-1, keepdims=True)
        normed_target_embs = target_embs / np.linalg.norm(target_embs, ord=2, axis=-1, keepdims=True)
        return normed_source_embs @ normed_target_embs.T

    def preprocess(self, inputs, topk=8, m=40):
        inputs = self.custom_processor(inputs).to(self.device)
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        with torch.no_grad():
            image_features = self.vis_encoder.encode_image(inputs).cpu().numpy()
        self.features = image_features.copy()
        image_features = (image_features - self.coco_mean) / self.coco_std
        image_features = image_features @ self.projection

        sims = self._calc_cos_sims(image_features, self.targets)
        ranked_sims = np.argsort(sims, axis=-1)[:, ::-1]
        # deduplicate tokens
        dec = [self.tokenizer.decode([tok]).strip().lower() for tok in ranked_sims[0][:50]]
        unique, idx = np.unique(dec, return_index=True)
        top_toks = ranked_sims[0, :50][np.sort(idx)][:topk]

        insert_token = self.tokenizer.encode(' ')
        # delete bos token
        del insert_token[0]

        permuted_prompts = [top_toks]
        np.random.seed(101)
        for _ in np.arange(m - 1):
            perm = np.random.permutation(np.arange(topk))
            permuted_prompts.append(top_toks[perm])
        prefix = np.array(permuted_prompts)
        prefix = np.insert(prefix, np.arange(1, topk + 1), insert_token * topk, axis=-1)
        # add (llama) bos-token at beginning of each prompt
        prompt = self.tokenizer.encode("A picture of")[1:]
        prompt = np.array([prompt] * m)
        model_input = np.concatenate([prefix, prompt], axis=-1)

        return {"inds": torch.LongTensor(model_input).to(self.device)}

    def _forward(self, model_inputs, length=67, m=40):
        out = torch.zeros((m, 67)).to(self.device)
        generated = self.model.model.embed_tokens(model_inputs["inds"]).squeeze()
        tokens = None
        stop_tokens = ['.', '\n']
        stop_token_index = [self.tokenizer.encode(stop_token)[-1] for stop_token in stop_tokens]
        stop_token_index = stop_token_index + [29889, 13]
        bs, seqlen, *_ = generated.shape
        terminated = np.full(shape=(bs,), fill_value=False, dtype=np.bool8)

        for i in range(length):

            outputs = self.model(inputs_embeds=generated)

            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            terminated[np.in1d(next_token.cpu().numpy(), stop_token_index)] = True

            next_token_embed = self.model.model.embed_tokens(next_token)

            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=0)
            generated = torch.cat((generated, next_token_embed.transpose(0, 1)), dim=1)

            if terminated.any():
                dones = np.nonzero(terminated)[0]
                for done in dones:
                    l = len(tokens[:, done])
                    out[done][:l] = tokens[:, done]

                generated = generated[~terminated]
                tokens = tokens[:, ~terminated]
                if not len(generated):
                    # All beams are terminated
                    break
                terminated = np.full(shape=(generated.shape[0],), fill_value=False, dtype=np.bool8)

        if len(generated) > 0:
            dones = np.nonzero(~terminated)[0]
            for done in dones:
                out[done] = tokens[:, done]

        return out

    def postprocess(self, model_outputs):
        tokens = model_outputs.cpu().numpy()
        caps = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        caps = np.unique(caps)
        caps = [cap for cap in caps if len(cap)]
        tokenized = clip.tokenize(caps, truncate=True).to(self.device)
        with torch.no_grad():
            text_embs = self.vis_encoder.encode_text(tokenized).cpu().numpy()
        sims = self._calc_cos_sims(self.features, text_embs)
        cap = caps[np.argmax(sims)]
        return cap


PIPELINE_REGISTRY.register_pipeline(
    "sitta-image-to-text",
    pipeline_class=SITTA,
    pt_model=LlamaForCausalLM,
    default={'pt': ("decapoda-research/llama-7b-hf", '47dc237k')},
)


def main():
    options = parse_args()
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", cache_dir="/system/user/publicdata/llm")
    model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", cache_dir="/system/user/publicdata/llm")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device, index=0)
    sitta = pipeline("sitta-image-to-text", model=model, tokenizer=tokenizer, device=device)
    test_sample = Image.open(options.path)
    generated_cap = sitta(test_sample)
    print(generated_cap)


if __name__ == '__main__':
    main()

