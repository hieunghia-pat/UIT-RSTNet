import random

from data_utils.dataset import FeatureDataset, DictionaryDataset
from data_utils.vocab import Vocab
from data_utils.utils import collate_fn

from models.rstnet.language_model import LanguageModel

from evaluation import compute_language_scores

import torch
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm

import argparse
import os
import pickle
import numpy as np
from shutil import copyfile

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_loss(dataloader: data.DataLoader):
    # Calculating validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - Validation' % (epoch + 1), unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, sample in enumerate(dataloader):
                tokens = sample["tokens"].to(device)
                shifted_right_tokens = sample["shifted_right_tokens"].to(device)
                out, _ = model(tokens)
                out = out.contiguous()
                loss = loss_fn(out.view(-1, len(vocab)), shifted_right_tokens.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)

    return val_loss


def evaluate_metrics(dataloader: data.DataLoader):
    model.eval()
    with tqdm(desc='Epoch %d - Evaluation' % (epoch + 1), unit='it', total=len(dataloader)) as pbar:
        for it, sample in enumerate(dataloader):
            gt_ids = sample["tokens"].to(device)
            with torch.no_grad():
                out, _ = model(gt_ids)
                out = out.contiguous()
            predicted_ids = out.argmax(dim=-1)
            captions_gt = vocab.decode_caption(gt_ids, join_words=False)
            captions_gen = vocab.decode_caption(predicted_ids, join_words=False)
            scores = compute_language_scores(captions_gt, captions_gen)
            pbar.update()

    return scores

def train_xe():
    # Training with cross-entropy loss
    model.train()

    running_loss = .0
    with tqdm(desc='Epoch %d - Training with cross-entropy loss' % (epoch + 1), unit='it', total=len(train_dataloader)) as pbar:
        for it, sample in enumerate(train_dataloader):
            tokens = sample["tokens"].to(device)
            shifted_right_tokens = sample["shifted_right_tokens"].to(device)
            out, _ = model(tokens)
            out = out.contiguous()
            optim.zero_grad()
            loss = loss_fn(out.view(-1, len(vocab)), shifted_right_tokens.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Bert Language Model')
    parser.add_argument('--exp_name', type=str, default='bert_language')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=11328)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')

    parser.add_argument('--features_path', type=str, default='./Datasets/X101-features/X101-grid-coco_trainval.hdf5')
    parser.add_argument('--train_json_path', type=str, default='features/annotations/UIT-ViIC/uitviic_captions_train2017.json')
    parser.add_argument('--val_json_path', type=str, default='features/annotations/UIT-ViIC/uitviic_captions_train2017.json')
    parser.add_argument('--test_json_path', type=str, default='features/annotations/UIT-ViIC/uitviic_captions_train2017.json')
    
    parser.add_argument('--dir_to_save_model', type=str, default='./saved_language_models')
    
    args = parser.parse_args()

    print('PhoBert Language Model Training')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # creating checkpoint directory
    if not os.path.isdir(os.path.join(args.dir_to_save_model, 
                                        f"{args.exp_name}")):
        os.makedirs(os.path.join(args.dir_to_save_model, 
                                    f"{args.exp_name}"))

    # Creating vocabulary and dataset
    if not os.path.isfile(os.path.join(args.dir_to_save_model, 
                                        f"{args.exp_name}", "vocab.pkl")):
        print("Creating vocab ...")
        vocab = Vocab([args.train_json_path, args.val_json_path], tokenizer_name="vncorenlp", 
                        pretrained_language_model_name="vinai/phobert-base")
        pickle.dump(vocab, open(os.path.join(args.dir_to_save_model, 
                                        f"{args.exp_name}", "vocab.pkl"), "wb"))
    else:
        print("Loading vocab ...")
        vocab = pickle.load(open(os.path.join(args.dir_to_save_model, 
                                        f"{args.exp_name}", "vocab.pkl"), "rb"))

    # creating iterable dataset
    print("Creating datasets ...")
    train_dataset = FeatureDataset(args.train_json_path, args.features_path, vocab) # for training with cross-entropy loss
    val_dataset = FeatureDataset(args.val_json_path, args.features_path, vocab) # for training with cross-entropy loss
    test_dataset = FeatureDataset(args.test_json_path, args.features_path, vocab) # for training with cross-entropy loss

    # creating dictionary dataset
    train_dict_dataset = DictionaryDataset(args.train_json_path, args.features_path, vocab) # for training with self-critical learning
    val_dict_dataset = DictionaryDataset(args.val_json_path, args.features_path, vocab) # for calculating metricsn validation set
    test_dict_dataset = DictionaryDataset(args.test_json_path, args.features_path, vocab=vocab)

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn
    )
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn
    )

    model = LanguageModel(vocab=vocab, padding_idx=vocab.padding_idx, bert_hidden_size=768, vocab_size=len(vocab)).to(device)

    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        if s % 11331 == 0:
            s = 1
        else:
            s = s % 11331

        lr = (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
        if lr > 1e-6:
            lr = 1e-6

        return lr

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    # scheduler = StepLR(optim, step_size=2, gamma=0.5)
    loss_fn = NLLLoss(ignore_index=vocab.padding_idx)
    use_rl = False
    best_score = .0
    best_test_score = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name)
        else:
            fname = os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name)

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_score = data['best_score']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best score %f' % (
                data['epoch'], data['val_loss'], data['best_score']))

    print("Training starts")
    for epoch in range(start_epoch, start_epoch + 100):
        if not use_rl:
            train_loss = train_xe()
        else:
            break

        # Validation loss
        val_loss = evaluate_loss(val_dataloader)

        # Validation scores
        val_scores = evaluate_metrics(val_dataloader)
        print(f"epoch {epoch+1}: Validation scores", val_scores)
        val_score = val_scores['f1']

        # Test scores
        test_scores = evaluate_metrics(test_dataloader)
        print(f"epoch {epoch+1}: Test scores", test_scores)
        test_score = test_scores['f1']

        # Prepare for next epoch
        best = False
        if val_score >= best_score:
            best_score = val_score
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_score >= best_test_score:
            best_test_score = test_score
            best_test = True

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                break
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load(os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best score %f' % (
                data['epoch'], data['val_loss'], data['best_score']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_score': val_score,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_score': best_score,
            'use_rl': use_rl,
        }, os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name))

        if best:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
        if best_test:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best_test.pth' % args.exp_name))

        if exit_train:
            break
