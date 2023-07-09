from argparse import ArgumentParser
import numpy as np
np.random.seed(101)
from sklearn.linear_model import Ridge, LinearRegression
import os
import tqdm
from sklearn.model_selection import KFold
from utils import orthogonal_procrustes, calc_cosine_sim


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--lm', default='llama-7b-hf', help='Language model embeddings to take')
    return parser.parse_args()


def accuracy_at_1(sims):
    if len(sims.shape) == 2:
        sims = sims.argmax(-1)
    acc = np.sum(sims == np.arange(len(sims))) / len(sims)
    return acc


def accuracy_at_5(sims):
    top5_sims = np.argsort(sims, axis=-1)[:, -5:]
    counts = 0
    for i, pred in enumerate(top5_sims):
        if i in pred:
            counts += 1
    acc = counts / len(sims)
    return acc


def perform_cv(src_embs, target_embs, train_method, eval_metric):
    cv = KFold(n_splits=5, shuffle=True, random_state=101)
    train_accs1 = []
    test_accs1 = []

    for i, (train_inds, test_inds) in enumerate(cv.split(src_embs)):

        train_src = src_embs[train_inds]
        train_tar = target_embs[train_inds]

        # fit linear model
        if train_method == 'procrustes':
            proj_mat = orthogonal_procrustes(train_src, train_tar)
        elif train_method == 'robust_procrustes':
            # Robust procrustes method from https://arxiv.org/abs/2205.11616
            eps = 1e-3
            m = 5
            proj_mat = orthogonal_procrustes(train_src, train_tar)
            for j in range(m):
                weights = 1 / (np.linalg.norm(train_tar - (train_src @ proj_mat), ord=2, axis=-1) + eps)
                weights = weights / np.max(weights)
                diag = np.diag(weights**0.5)
                proj_mat = orthogonal_procrustes(diag @ train_src, diag @ train_tar)
        elif train_method == 'linear_reg':
            model = LinearRegression(fit_intercept=False)
            model.fit(train_src, train_tar)
            proj_mat = model.coef_.T
        elif train_method == 'ridge_reg':
            model = Ridge(alpha=1., fit_intercept=False)
            model.fit(train_src, train_tar)
            proj_mat = model.coef_.T
        else:
            raise NotImplementedError(f'{train_method} - Training method not supported!!')

        # evaluate current fold
        train_cosine_sims = calc_cosine_sim(train_src, train_tar, proj_mat)
        train_acc1 = eval_metric(train_cosine_sims)
        train_accs1.append(train_acc1)

        test_src = src_embs[test_inds]
        test_tar = target_embs[test_inds]

        test_cosine_sims = calc_cosine_sim(test_src, test_tar, proj_mat)
        test_acc1 = eval_metric(test_cosine_sims)
        test_accs1.append(test_acc1)

    print("--------------------------------------------------\n")
    print(f"Average over 5 Folds, {train_method}\n")
    print(f"Train Accuracy@1: {np.mean(train_accs1)}\n")
    print(f"Test Accuracy@1: {np.mean(test_accs1)}\n")
    print("--------------------------------------------------\n")


def get_label_for_embs(embs, targets, batch_size=1024):
    new_targets = []
    for i in tqdm.trange(0, len(embs), batch_size, desc='Retrieving new labels for remaining tokens...'):
        # compute cosine similarity
        sims = embs[i:i+batch_size] @ targets.T
        inds = np.argmax(sims, axis=-1)
        new_targets.extend(targets[inds].tolist())
    return np.array(new_targets)


def main():
    options = create_parser()
    os.makedirs('../models', exist_ok=True)

    clip_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B32', 'ViT-B16', 'ViT-L14',
                    'ViT-L14@336px']


    lm = options.lm
    for encoder in clip_models:

        clip_embs = np.load(f'./data/{encoder}_{lm}_prompt_embs.npz')
        target_embs = np.load(f'./data/{lm}_embs.npz')

        # center and scale data
        clip_mean = clip_embs.mean(0)
        clip_std = clip_embs.std(0)
        target_mean = target_embs.mean(0)
        target_std = target_embs.std(0)
        clip_embs = (clip_embs - clip_mean) / clip_std
        target_embs = (target_embs - target_mean) / target_std
        print(f"Aligning {len(clip_embs)} Tokens")
    
        for train_method in ['linear_reg', 'ridge_reg', 'procrustes', 'robust_procrustes']:

            if not os.path.exists(os.path.join('../models', f'{lm}_{encoder}_{train_method}.npy')):
                # By default perform the token classification task to evaluate alignment
                perform_cv(clip_embs, target_embs, train_method, accuracy_at_1)

                # train final model
                if train_method == 'procrustes':
                    proj_mat = orthogonal_procrustes(clip_embs, target_embs)
                elif train_method == 'robust_procrustes':
                    # Robust procrustes method from https://arxiv.org/abs/2205.11616
                    eps = 1e-3
                    m = 5
                    proj_mat = orthogonal_procrustes(clip_embs, target_embs)
                    for j in range(m):
                        weights = 1 / (np.linalg.norm(target_embs - (clip_embs @ proj_mat), ord=2,
                                                      axis=-1) + eps)
                        weights = weights / np.max(weights)
                        diag = np.diag(weights ** 0.5)
                        proj_mat = orthogonal_procrustes(diag @ clip_embs, diag @ target_embs)
                elif train_method == 'linear_reg':
                    model = LinearRegression(fit_intercept=False)
                    model.fit(clip_embs, target_embs)
                    proj_mat = model.coef_.T
                elif train_method == 'ridge_reg':
                    model = Ridge(alpha=1., fit_intercept=False)
                    model.fit(clip_embs, target_embs)
                    proj_mat = model.coef_.T

                np.save(os.path.join('../models', f'{lm}_{encoder}_{train_method}'), proj_mat)
            else:
               print(f"{lm}_{encoder}_{train_method} - Mapping already exists!!!")

if __name__ == '__main__':
    main()
