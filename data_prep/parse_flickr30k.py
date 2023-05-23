from PIL import Image
import json
import os
from collections import defaultdict
import pickle
import tqdm


def extract_img_and_caps(root, data, split_dict):
    imgs = {}
    txts = defaultdict(str)
    imgs_train = {}
    txts_train = defaultdict(str)
    imgs_test = {}
    txts_test = defaultdict(str)
    imgs_val = {}
    txts_val = defaultdict(str)

    for i in tqdm.trange(len(data), desc="Loading Flickr30k data..."):
        d = data[i]
        img_id = d["image_id"]
        filename = f"{root}/{img_id}.jpg"

        try:
            split = split_dict[img_id]
        except KeyError:
            continue

        image = Image.open(filename)
        imgs[filename] = image.copy()
        txts[filename] += f' {d["caption"]}'

        if split == 'train':
            imgs_train[filename] = image.copy()
            txts_train[filename] += f' {d["caption"]}'
        elif split == 'test':
            imgs_test[filename] = image.copy()
            txts_test[filename] += f' {d["caption"]}'
        else:
            imgs_val[filename] = image.copy()
            txts_val[filename] += f' {d["caption"]}'

    return imgs, txts, imgs_train, txts_train, imgs_test, txts_test, imgs_val, txts_val


def main():
    annotations = 'anns_flickr30k.json'
    split_annotations = 'dataset_flickr30k.json'
    root = './datasets/flickr30k'

    with open(os.path.join('./annotations', annotations), 'r') as f:
        data = json.load(f)

    with open(os.path.join(root, split_annotations), 'r') as f:
        split_data = json.load(f)
        split_dict = {d['filename'].split('.')[0]: d['split'] for d in split_data['images']}

    os.makedirs(os.path.join('../data', 'flickr30k'), exist_ok=True)

    imgs, txts, imgs_train, txts_train, imgs_test, txts_test, imgs_val, txts_val = extract_img_and_caps(root, data, split_dict)

    pickle.dump(txts_train, open(os.path.join('../data', 'flickr30k', f'txts_val.pkl'), 'wb'))
    pickle.dump(imgs_train, open(os.path.join('../data', 'flickr30k', f'imgs_val.pkl'), 'wb'))

    pickle.dump(txts_val, open(os.path.join('../data', 'flickr30k', f'txts_val.pkl'), 'wb'))
    pickle.dump(imgs_val, open(os.path.join('../data', 'flickr30k', f'imgs_val.pkl'), 'wb'))

    pickle.dump(txts_test, open(os.path.join('../data', 'flickr30k', f'txts_val.pkl'), 'wb'))
    pickle.dump(imgs_test, open(os.path.join('../data', 'flickr30k', f'imgs_val.pkl'), 'wb'))

    print('Done')
    return 0


if __name__ == '__main__':
    exit(main())
