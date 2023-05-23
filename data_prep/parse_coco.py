from PIL import Image
import json
import os
from collections import defaultdict
import pickle
import tqdm
from argparse import ArgumentParser


def extract_img_and_caps(root, data):
    imgs = {}
    txts = defaultdict(str)

    for i in tqdm.trange(len(data), desc="Loading COCO data..."):
        d = data[i]
        filename = f"{root}/{d['filename']}"

        image = Image.open(filename)
        imgs[filename] = image.copy()
        txts[filename] += ' '.join([sent['raw'] for sent in d['sentences']])

    return imgs, txts


def main():
    annotations = '../dataset/mscoco/dataset_coco.json'

    with open(annotations, 'r') as f:
        data = json.load(f)['images']
        train_anns = [ann for ann in data if ann['split'] == 'train']
        val_anns = [ann for ann in data if ann['split'] == 'val']
        test_anns = [ann for ann in data if ann['split'] == 'test']

    os.makedirs(os.path.join('../data', 'mscoco'), exist_ok=True)

    root = './datasets/mscoco'
    imgs, txts = extract_img_and_caps(root, train_anns)
    pickle.dump(txts, open(os.path.join('../data', 'coco', f'txts_train.pkl'), 'wb'))
    pickle.dump(imgs, open(os.path.join('../data', 'coco', f'imgs_train.pkl'), 'wb'))

    imgs, txts = extract_img_and_caps(root, val_anns)
    pickle.dump(txts, open(os.path.join('../data', 'coco', f'txts_val.pkl'), 'wb'))
    pickle.dump(imgs, open(os.path.join('../data', 'coco', f'imgs_val.pkl'), 'wb'))

    imgs, txts = extract_img_and_caps(root, test_anns)
    pickle.dump(txts, open(os.path.join('../data', 'coco', f'txts_test.pkl'), 'wb'))
    pickle.dump(imgs, open(os.path.join('../data', 'coco', f'imgs_test.pkl'), 'wb'))

    print('Done')
    return 0


if __name__ == '__main__':
    exit(main())
