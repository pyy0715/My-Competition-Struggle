import os 
import sys 
import pickle
import pandas as pd 
from tqdm import tqdm
from util import iterate_data_files

def read_files_to_dataframe(dest_path):
   print('read files of all users')
   os.makedirs(dest_path, exist_ok=True)
   data = []
   files = sorted([path for path, _ in iterate_data_files('2018100100', '2019022200')])
   for path in tqdm(files, mininterval=1):
      read_datetime = path.split('/')[-1].split('_')[0][:9]
      for line in open(path):
         tokens = line.strip().split()
         user_id = tokens[0]
         reads = tokens[1:]
         for item in reads:
            data.append([read_datetime, user_id, item])
   read_df = pd.DataFrame(data)
   read_df.columns = ['date', 'user_id', 'article_id']
   read_df.to_csv(os.path.join(dest_path, 'read_df.csv'), index=False, encoding='utf-8-sig')
   
def user_seen_article(src_path, dest_path):
   read_df = pd.read_csv(src_path, parse_dates=['date'])
   
   train_period = (read_df.query('date<"20190222"'))
   dev_period = (read_df.query('date>="20190222"'))
   
   seen_article = train_period.groupby('user_id')['article_id'].apply(set)
   dev_seen_article = dev_period.groupby('user_id')['article_id'].apply(set)
   
   with open(os.path.join(dest_path, 'seen_article.pkl'), 'wb') as f:
      pickle.dump(seen_article, f)
   with open(os.path.join(dest_path, 'dev_seen_article.pkl'), 'wb') as f:
      pickle.dump(dev_seen_article, f)
   
if __name__ == '__main__':
   DEST_PATH = '../assets/'
   DATA = 'read_df.csv'
   SRC_PATH = os.path.join(DEST_PATH, DATA)
   
   read_files_to_dataframe(DEST_PATH)
   user_seen_article(SRC_PATH, DEST_PATH)
   
   
   
   
   
    
   
