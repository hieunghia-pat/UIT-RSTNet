import random

from data_utils.dataset import FeatureDataset, DictionaryDataset
from data_utils.vocab import Vocab
from data_utils.utils import collate_fn
import evaluation
from evaluation import PTBTokenizer, Cider

from models.rstnet import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention

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
import itertools
import multiprocessing
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
                    features = sample["grid_features"].to(device)
                    captions = sample["tokens"].to(device)
                    out = model(features, captions)
                    captions = captions[:, 1:].contiguous()
                    out = out[:, :-1].contiguous()
                    loss = loss_fn(out.view(-1, len(vocab)), captions.view(-1))
                    this_loss = loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss

def evaluate_metrics(dataloader: data.DataLoader):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - Evaluation' % (epoch + 1), unit='it', total=len(dataloader)) as pbar:
        for it, sample in enumerate(dataloader):
            images = sample["grid_features"].to(device)
            caps_gt = sample["captions"]
            with torch.no_grad():
                out, _ = model.beam_search(images, vocab.max_caption_length, vocab.eos_idx, 5, out_size=1)
            caps_gen = vocab.decode_caption(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    
    return scores

def train_xe():
    # Training with cross-entropy loss
    model.train()

    running_loss = .0
    with tqdm(desc='Epoch %d - Training with cross-entropy loss' % (epoch + 1), unit='it', total=len(train_dataloader)) as pbar:
        for it, sample in enumerate(train_dataloader):
            features = sample["grid_features"].to(device)
            captions = sample["tokens"].to(device)
            out = model(features, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            # scheduler.step()

    loss = running_loss / len(train_dataloader)
    return loss

def train_scst():
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0

    model.train()
    scheduler_rl.step()
    print('lr = ', optim_rl.state_dict()['param_groups'][0]['lr'])

    running_loss = .0
    seq_len = vocab.max_caption_length
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_dict_dataloader)) as pbar:
        for it, sample in enumerate(train_dict_dataloader):
            detections = sample["grid_features"].to(device)
            outs, log_probs = model.beam_search(detections, seq_len, vocab.eos_idx,
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = vocab.decode_caption(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = train_cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(train_dict_dataloader)
    reward = running_reward / len(train_dict_dataloader)
    reward_baseline = running_reward_baseline / len(train_dict_dataloader)
    return loss, reward, 


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='RSTNet')
    parser.add_argument('--exp_name', type=str, default='rstnet')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')

    parser.add_argument('--features_path', type=str, default='./Datasets/X101-features/X101-grid-coco_trainval.hdf5')
    parser.add_argument('--train_json_path', type=str, default='features/annotations/UIT-ViIC/uitviic_captions_train2017.json')
    parser.add_argument('--val_json_path', type=str, default='features/annotations/UIT-ViIC/uitviic_captions_train2017.json')
    parser.add_argument('--test_json_path', type=str, default='features/annotations/UIT-ViIC/uitviic_captions_train2017.json')

    parser.add_argument('--dir_to_save_model', type=str, default='./saved_transformer_models/')
    
    parser.add_argument('--xe_least', type=int, default=15)
    parser.add_argument('--xe_most', type=int, default=20)
    parser.add_argument('--refine_epoch_rl', type=int, default=28)

    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)

    args = parser.parse_args()

    print('The Training of RSTNet')
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

    train_dict_dataloader = data.DataLoader(
            dataset=train_dict_dataset,
            batch_size=args.batch_size // 5,
            shuffle=True,
            collate_fn=collate_fn
        )
    val_dict_dataloader = data.DataLoader(
        dataset=val_dict_dataset,
        batch_size=args.batch_size // 5,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dict_dataloader = data.DataLoader(
        dataset=test_dict_dataset,
        batch_size=args.batch_size // 5,
        shuffle=True,
        collate_fn=collate_fn
    )

    train_cider = Cider(PTBTokenizer.tokenize(train_dataset.captions))

    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': args.m})
    decoder = TransformerDecoderLayer(len(vocab), vocab.max_caption_length, 3, vocab.padding_idx)
    model = Transformer(vocab.bos_idx, encoder, decoder).to(device)

    def lambda_lr(s):
        print("s:", s)
        if s <= 3:
            lr = args.xe_base_lr * s / 4
        elif s <= 10:
            lr = args.xe_base_lr
        elif s <= 12:
            lr = args.xe_base_lr * 0.2
        else:
            lr = args.xe_base_lr * 0.2 * 0.2
        return lr
    
    def lambda_lr_rl(s):
        refine_epoch = args.refine_epoch_rl 
        print("rl_s:", s)
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 3:
            lr = args.rl_base_lr * 0.2
        elif s <= refine_epoch + 6:
            lr = args.rl_base_lr * 0.2 * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
        return lr


    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

    loss_fn = NLLLoss(ignore_index=vocab.padding_idx)
    use_rl = False
    best_cider = .0
    best_test_cider = 0.
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
            """
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            """
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            use_rl = data['use_rl']

            if use_rl:
                optim.load_state_dict(data['optimizer'])
                scheduler.load_state_dict(data['scheduler'])
            else:
                optim_rl.load_state_dict(data['optimizer'])
                scheduler_rl.load_state_dict(data['scheduler'])

            print('Resuming from epoch %d, validation loss %f, best cider %f, and best_test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))
            print('patience:', data['patience'])

    print("Training starts")
    for epoch in range(start_epoch, start_epoch + 100):
        if not use_rl:
            train_loss = train_xe()
        else:
            train_loss, reward, reward_baseline = train_scst()

        # Validation loss
        val_loss = evaluate_loss(val_dataloader)

        # Validation scores
        scores = evaluate_metrics(val_dict_dataloader)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']

        # Test scores
        scores = evaluate_metrics(test_dict_dataloader)
        print("Test scores", scores)
        test_cider = scores['CIDEr']

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False

        if patience == 5:
            if epoch < args.xe_least:   # xe stage train 15 epoches at least 
                print('special treatment, e = {}'.format(epoch))
                use_rl = False
                switch_to_rl = False
                patience = 0
            elif not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
                
                for k in range(epoch-1):
                    scheduler_rl.step()

                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if epoch == args.xe_most:     # xe stage no more than 20 epoches
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

                for k in range(epoch-1):
                    scheduler_rl.step()

                print("Switching to RL")

        if switch_to_rl and not best:
            data = torch.load(os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, best_cider %f, and best test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
            'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'best_test_cider': best_test_cider,
            'use_rl': use_rl,
        }, os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name))
        
        if best:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
        if best_test:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best_test.pth' % args.exp_name))

        if exit_train:
            break
