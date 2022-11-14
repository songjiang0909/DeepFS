from pickle import TRUE
import time
import argparse
import torch
from utils import *
from data import *
from model import DeepFS
from exp import Experiments
from datetime import datetime
import pytz

print ("Start time:")
tz_LA = pytz.timezone("America/Los_Angeles")
datetime_LA = datetime.now(tz_LA)
print("LA time:", datetime_LA.strftime("%D  %H:%M"))


parser = argparse.ArgumentParser()

##data and simulation
parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset') #[simulation,"ETTh1","ETTh2","ETTm1","ETTm2","WTH"]
parser.add_argument('--is_full', type=bool, default=True, help='data simulation: pure periodic or plus trend and noise')
parser.add_argument('--is_linear', type=bool, default=False, help='Linear or ploy non-periodic')
parser.add_argument('--numOfBase', type=int, default=50, help='number Of periods base in simulation, not model')
parser.add_argument('--numOfSamples', type=int, default=40000, help='number Of data samples in simulation')
parser.add_argument('--granularity', type=int, default=1, help='granularity of termporl position in simulation')
parser.add_argument('--k', type=float, default=0.000001, help='parameter for non-linear simulation-k')
parser.add_argument('--b', type=float, default=0.5, help='parameter for non-linear simulation-b')
parser.add_argument('--scale', type=int, default=1, help='normalize input data')

parser.add_argument('--train_rate', type=float, default=0.7, help='training rate')
parser.add_argument('--val_rate', type=float, default=0.1, help='validation rate')

parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--input_len', type=int, default=720, help='input sequence length')
parser.add_argument('--out_len', type=int, default=168, help='output sequence length')
parser.add_argument('--embed_size', type=int, default=100, help='dimension of embeddings')
parser.add_argument('--model_selection', type=str, default="DeepFS", help='which model is using')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--distil', action='store_true', help='whether to use distilling in encoder, using this argument means not using distilling')
parser.add_argument('--base', type=int, default=100, help='number of periodic bases')
parser.add_argument('--inverse', action='store_true', help='inverse output data')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--activation', type=str, default='gelu',help='activation')

parser.add_argument('--expID', type=int, default=0, help='-th of experiments')
parser.add_argument('--epochs', type=int, default=50, help='training epochs')
parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--alpha', type=float, default=1, help='weight of period signals')
parser.add_argument('--beta', type=float, default=1, help='weight of non-period signals')
parser.add_argument('--patience', type=int, default=5, help='early stopping')
parser.add_argument('--delta', type=float, default=0, help='early stopping tolerance')
parser.add_argument('--start_watch', type=int, default=20, help='epoch to start early stopping')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda for training')


parser.add_argument('--no_stop', type=int, default=0, help='whether use early stop, for debugging')



startTime = time.time()
args = parser.parse_args()
args.cuda = True if torch.cuda.is_available() and args.cuda else False

args.is_sim = 1 if args.dataset=="simulation" else 0



data_parser = {
    'simulation':{'data_path':'simulation_'+str(args.numOfBase)+'_'+str(args.numOfSamples)+'_'+str(args.granularity)+'.csv','T':'V','S':[1,1,1]},
    'ETTh1':{'data_path':'ETTh1.csv','T':'OT','S':[1,1,1]},
    'ETTh2':{'data_path':'ETTh2.csv','T':'OT','S':[1,1,1]},
    'ETTm1':{'data_path':'ETTm1.csv','T':'OT','S':[1,1,1]},
    'ETTm2':{'data_path':'ETTm2.csv','T':'OT','S':[1,1,1]},
    'WTH':{'data_path':'WTH.csv','T':'WetBulbCelsius','S':[1,1,1]},
}
if args.dataset in data_parser.keys():
    data_info = data_parser[args.dataset]
    args.data_path = data_info['data_path']
    args.target = data_info['T']
print (args)

    
train_data, train_loader = load_data(args,flag = 'train')
val_data, val_loader = load_data(args,flag = 'val')
test_data, test_loader = load_data(args,flag = 'test')



model_dict = {
    'DeepFS':DeepFS,
}
model = model_dict[args.model_selection](args)
exp = Experiments(args,model,train_loader,val_loader,test_loader,train_data,val_data,test_data)


"""Train the model"""
exp.train()

"""Save results"""
exp.save_prediction()


print("Time usage:{:.4f} mins".format((time.time()-startTime)/60))
tz_LA = pytz.timezone("America/Los_Angeles")
datetime_LA = datetime.now(tz_LA)
print("LA time:", datetime_LA.strftime("%D  %H:%M"))
print ("****************************THE END********************************")


