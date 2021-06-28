import os
import gc
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import easydict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from train import build_fn
from data import AudioDataset

args = easydict.EasyDict({'sr': 16000,
                          'n_mels': 128,
                          'n_fft': [1024],
                          'win_length': [600],
                          'hop_length': 120,
                          'min_length': 120000,
                          'min_level_db': -80,
                          'lr': 1e-4,
                          'epochs': 20,
                          'seed': 2021,
                          'batch_num': 128,
                          'fp16': True
                          })

data_dir = './dataset'
train_npy_dir = './dataset/train_npy'
test_npy_dir = './dataset/test_npy'

if __name__ == '__main__':
   pl.seed_everything(args['seed'])
   train_df = pd.read_pickle(os.path.join(data_dir, 'new_train.pkl'))
   test_df = pd.read_pickle(os.path.join(data_dir, 'test.pkl'))
   
   skf = StratifiedKFold(n_splits=5, random_state=args['seed'], shuffle=True)

   for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_df.values, train_df['accent'])):
      trn_df, val_df = train_df.iloc[trn_idx], train_df.iloc[val_idx]

      train_ds = AudioDataset(args, trn_df, transform=True)
      valid_ds = AudioDataset(args, val_df, transform=False)

      train_loader = torch.utils.data.DataLoader(
         train_ds, batch_size=args['batch_num'], shuffle=True, num_workers=4, pin_memory=True)
      valid_loader = torch.utils.data.DataLoader(
         valid_ds, batch_size=args['batch_num'], shuffle=False, num_workers=4, pin_memory=False)

      checkpoint_callback = ModelCheckpoint(
         filename='{epoch}-{val_acc:.2f}-{val_loss:.3f}',
         monitor='val_loss',
         save_top_k=1,
         mode='min')

      early_stop_callback = EarlyStopping(monitor='val_loss',
                                          patience=4,
                                          verbose=True,
                                          mode='min')

      trainer = pl.Trainer(
         callbacks=[checkpoint_callback, early_stop_callback],
         max_epochs=args['epochs'],
         deterministic=torch.cuda.is_available(),
         gpus=-1 if torch.cuda.is_available() else None,
         precision=16 if args['fp16'] else 32)

      pl_model = build_fn(args, train_loader, valid_loader)

      trainer.fit(pl_model)
      
