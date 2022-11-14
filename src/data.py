import os
import time
import pandas as pd
import torch
from simulation import *
from utils import StandardScaler
from timefeatures import time_features
from torch.utils.data import Dataset, DataLoader



class Dataset_Custom(Dataset):
    """Codes are adapted from https://github.com/zhouhaoyi/Informer2020."""
    
    def __init__(self, args,data_path, flag='train',features='S', 
                 target='V', inverse=False, timeenc=0, freq='h', cols=None):
        # info
        self.args = args
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.input_len = args.input_len
        self.out_len = args.out_len
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = args.scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join("../data/",
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature, time stamp]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
            cols.remove("time")
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date'); cols.remove('time')
        df_raw = df_raw[['date']+cols+[self.target]+["time"]]

        num_train = int(len(df_raw)*self.args.train_rate)
        num_val = int(len(df_raw)*self.args.val_rate)
        num_test = len(df_raw) - num_train - num_val
        border1s = [0, num_train-self.input_len, len(df_raw)-num_test-self.input_len]
        border2s = [num_train, num_train+num_val, len(df_raw)]

        if self.data_path in set(["ETTh1.csv","ETTh2.csv"]):
            print ("I'm ETTh")
            border1s = [0, 12*30*24 - self.input_len, 12*30*24+4*30*24 - self.input_len]
            border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]

        if self.data_path in set(["ETTm1.csv"]):
            print ("I'm ETTm")
            border1s = [0, 12*30*24*4 - self.input_len, 12*30*24*4+4*30*24*4 - self.input_len]
            border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        
        if self.data_path in set(["WTH.csv"]):
            print ("WTH")
            border1s = [0, 28*30*24 - self.input_len, 28*30*24+10*30*24 - self.input_len]
            border2s = [28*30*24, 28*30*24+10*30*24, 28*30*24+20*30*24]
        


        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.times = df_raw[['time']].values[border1:border2]

        self.data_stamp = data_stamp
    

    def __getitem__(self, index):
        s_begin = index
        # s_begin = 0
        s_end = s_begin + self.input_len
        r_begin = s_end 
        r_end = r_begin + self.out_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        timestamps = self.times[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, timestamps
    
    def __len__(self):
        return len(self.data_x) - self.input_len - self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def load_data(args, flag):

    timeenc = 0 if args.embed!='timeF' else 1

    if flag in set(['test','val']) :
        shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
    else:
        shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
    



    data_set = Dataset_Custom(
        args=args,
        data_path=args.data_path,
        flag=flag,
        # features=args.features,
        target=args.target,
        inverse=args.inverse,
        timeenc=timeenc,
        freq=freq,
        cols=args.cols
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)

    return data_set, data_loader


def simulate_data(args):
    assert (args.is_sim==True)
    signal_sim(args,args.is_full,args.is_linear,args.numOfBase,args.input_len,args.out_len,args.numOfSamples,args.granularity)
    time.sleep(5)
    print ("************ data simulation is done! ************")


