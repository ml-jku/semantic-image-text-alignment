from PIL import Image
import json
import os
from collections import defaultdict
import pickle
import tqdm


def extract_img_and_caps(root, data):
    imgs = {}
    txts = defaultdict(str)

    for i in tqdm.trange(len(data), desc="Loading Flickr30k data..."):
        d = data[i]
        img_id = d["image_id"]
        filename = f"{root}/{img_id}.jpg"

        image = Image.open(filename)
        imgs[filename] = image.copy()
        txts[filename] += f' {d["caption"]}'

    return imgs, txts


def main():
    split_annotations = 'dataset_flickr30k.json'

    with open(os.path.join('../../assets/annotations', split_annotations), 'r') as f:
        split_data = json.load(f)
        data = [{'caption': sent['raw'], 'image_id': img['filename'].split('.')[0]} for img in split_data['images'] for sent in img['sentences']]
        train_anns = [ann for ann in data if ann['split'] == 'train']
        val_anns = [ann for ann in data if ann['split'] == 'val']
        test_anns = [ann for ann in data if ann['split'] == 'test']

    os.makedirs(os.path.join('../data', 'flickr30k'), exist_ok=True)

    root = '../datasets/flickr30k'
    imgs_train, txts_train = extract_img_and_caps(root, train_anns)
    imgs_val, txts_val = extract_img_and_caps(root, val_anns)
    imgs_test, txts_test = extract_img_and_caps(root, test_anns)

    pickle.dump(txts_train, open(os.path.join('../data', 'flickr30k', f'txts_train.pkl'), 'wb'))
    pickle.dump(imgs_train, open(os.path.join('../data', 'flickr30k', f'imgs_train.pkl'), 'wb'))

    pickle.dump(txts_val, open(os.path.join('../data', 'flickr30k', f'txts_val.pkl'), 'wb'))
    pickle.dump(imgs_val, open(os.path.join('../data', 'flickr30k', f'imgs_val.pkl'), 'wb'))

    pickle.dump(txts_test, open(os.path.join('../data', 'flickr30k', f'txts_test.pkl'), 'wb'))
    pickle.dump(imgs_test, open(os.path.join('../data', 'flickr30k', f'imgs_test.pkl'), 'wb'))

    print('Done')
    return 0


if __name__ == '__main__':
    exit(main())
