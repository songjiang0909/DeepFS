from cgi import test
import time
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from torch import optim
from utils import *



class Experiments(object):
    def __init__(self,args,model,train_loader,val_loader,test_loader,train_data,val_data,test_data):
        super(Experiments,self).__init__()
        
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.optimizer = self._select_optimizer()
        self.loss = self._select_lossFc()
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=False,delta=self.args.delta)
        self.early_stop_sign = False

        self._set_device()
        self.exp_setting= str(self.args.dataset)+"_"+str(self.args.expID)+"_"+str(self.args.input_len)+"_"+str(self.args.out_len)+"_"+str(self.args.embed_size)+"_"+str(self.args.batch_size)\
            +"_"+str(self.args.epochs)+"_"+str(self.args.n_heads)+"_"+str(self.args.e_layers)+"_"+str(self.args.alpha)+"_"+str(self.args.beta)+"_"+str(self.args.lr)\
                +"_"+str(self.args.base)+"_"+str(self.args.patience)+"_"+str(self.args.delta)+"_"+str(self.args.start_watch)+"_"+str(self.args.no_stop)+"_"+str(self.args.numOfBase)
        if self.args.is_sim:
            self.exp_setting+=("_"+str(self.args.numOfSamples))
            self.exp_setting+=("_"+str(self.args.granularity))
        self.exp_setting+=("_"+str(self.args.model_selection))


    def _select_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer
    
    def _select_lossFc(self):
        loss = nn.MSELoss()

        return loss
    
    def _set_device(self):
        if self.args.cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()


    def train_one_step(self,batch_x, batch_y, batch_x_mark, batch_y_mark,timestamps):

        self.model.train()
        if self.args.cuda:
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float().cuda()
            batch_x_mark = batch_x_mark.float().cuda()
            timestamps = timestamps.float().cuda()

        self.optimizer.zero_grad()
        pred,_,_,_,_,_,_ = self.model(timestamps,batch_x,batch_x_mark)
        
        
        loss = self.loss(pred,batch_y)
        loss.backward()
        self.optimizer.step()

        return loss


    def train(self):
        
        time_tracker = []
        for epoch in range(self.args.epochs):
            epoch_time = time.time()
            train_loss = []
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,timestamps) in enumerate(self.train_loader):
                timestamps = timestamps.squeeze()
                batch_y = batch_y.squeeze()
                loss = self.train_one_step(batch_x, batch_y, batch_x_mark, batch_y_mark,timestamps)
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            val_loss = self.validation(self.val_loader)
            test_loss = self.validation(self.test_loader)


            time_tracker.append(time.time()-epoch_time)

            print('Epoch: {:04d}'.format(epoch + 1),
                ' train_loss:{:.06f}'.format(train_loss),
                ' val_loss:{:.06f}'.format(val_loss),
                ' test_loss:{:.06f}'.format(test_loss),
                ' alpha:{:.02f}'.format(self.args.alpha),
                ' beta:{:.02f}'.format(self.args.beta),
                ' epoch_time:{:.02f}s'.format(time.time()-epoch_time),
                ' remain_time:{:.02f}s'.format(np.mean(time_tracker)*(self.args.epochs-(1+epoch))),
                )

            if epoch+1>self.args.start_watch:
                self.early_stop_sign = True
                self.early_stopping(val_loss, self.model, "../result/models/"+self.exp_setting,epoch+1)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

        if self.args.no_stop or (not self.early_stop_sign):
            torch.save(self.model.state_dict(), "../result/models/"+self.exp_setting+'_'+'checkpoint.pth')


    def validation(self,val_loader):
        
        self.model.eval()
        val_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,timestamps) in enumerate(val_loader):
            
            if self.args.cuda:
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()
                timestamps = timestamps.float().cuda()
            
            timestamps = timestamps.squeeze()
            batch_y = batch_y.squeeze()
            
            pred,_,_,_,_,_,_ = self.model(timestamps,batch_x,batch_x_mark)
            loss = self.loss(pred,batch_y)
            val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

        return val_loss



    def test(self,data_loader,flag_label="test"):

        self.model.eval()
        test_loss = []
        test_pred = []
        test_gt = []
        test_weights = []
        test_phases = []
        test_period_pred = []
        test_n_period_pred = []
        ttt = []
        embeds = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,timestamps) in enumerate(data_loader):

            if self.args.cuda:
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()
                timestamps = timestamps.float().cuda()
            
            timestamps = timestamps.squeeze()
            batch_y = batch_y.squeeze()
                
            pred,weights_pred,periods_pred,phases_pred,period_pred,nperiod_pred,emb = self.model(timestamps,batch_x,batch_x_mark)
            loss = self.loss(pred,batch_y)
            test_loss.append(loss.item())
            test_pred.append(pred.detach().cpu().numpy())
            test_gt.append(batch_y.detach().cpu().numpy())
            test_weights.append(weights_pred.detach().cpu().numpy())
            test_phases.append(phases_pred.detach().cpu().numpy())
            test_period_pred.append(period_pred.detach().cpu().numpy())
            test_n_period_pred.append(nperiod_pred.detach().cpu().numpy())

            ttt.append(timestamps.detach().cpu().numpy())
            embeds.append(emb.detach().cpu().numpy())

        test_loss = np.average(test_loss)
        test_pred = np.concatenate(test_pred,0)
        if len(np.array(test_gt[-1]).shape) == 1:
            to_replace = np.array(test_gt[-1]).reshape(-1,len(test_gt[-1]))
            test_gt.pop()
            test_gt.append(to_replace)
        test_gt = np.concatenate(test_gt,0)
        test_weights = np.mean(test_weights,0)
        test_phases = np.mean(test_phases,0)
        if len(np.array(test_period_pred[-1]).shape) == 1:
            test_period_pred.pop()
            test_n_period_pred.pop()
            ttt.pop()
        test_period_pred = np.concatenate(test_period_pred,0)
        test_n_period_pred = np.concatenate(test_n_period_pred,0)
        ttt = np.concatenate(ttt,0)


        test_pred_recover = self.test_data.inverse_transform(test_pred)
        test_gt_recover = self.test_data.inverse_transform(test_gt)
        if flag_label=="test":
            mae,mse,rmse,mape,mspe = metric(test_pred, test_gt)
            print ("Normalized: ")
            print ("Error on test sets: mse: {:05f},mae: {:05f},rmse: {:05f},mape: {:05f},mspe: {:05f}".format(mse,mae,rmse,mape,mspe))
            print (" ")

            # mae,mse,rmse,mape,mspe = metric(test_pred_recover, test_gt_recover)
            # print ("After recovering")
            # print ("Error on test sets: mse: {:05f},mae: {:05f},rmse: {:05f},mape: {:05f},mspe: {:05f}".format(mse,mae,rmse,mape,mspe))


        return loss,test_pred,test_gt,test_weights,periods_pred,test_phases,test_period_pred,test_n_period_pred,test_pred_recover,test_gt_recover,ttt,embeds



    def save_prediction(self):
        
        print ("BEST EPOCH: {}".format(self.early_stopping.best_epoch))
        best_model_path = "../result/models/"+self.exp_setting+'_'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        self.model.eval()
        _,pred,gt,weights_pred,periods_pred,phases_pred,period_pred,n_period_pred,test_pred_recover,test_gt_recover,ttt,test_emb = self.test(self.test_loader)

        _,pred_train,gt_train,weights_pred_train,periods_pred_train,phases_pred_train,period_pred_train,n_period_pred_train,_,_,_,train_emb = self.test(self.train_loader,flag_label="train")

        
        data = {
            "input_len":self.args.input_len,
            "out_len":self.args.out_len,
            "pred":pred,
            "gt":gt,
            "weights":weights_pred,
            "periods":periods_pred,
            "phases":phases_pred,
            "period_pred":period_pred,
            "n_period_pred":n_period_pred,
            "pred_recover":test_pred_recover,
            "gt_recover":test_gt_recover,
            "ttt":ttt,


            "pred_train":pred_train,
            "gt_train":gt_train,
            "weights_train":weights_pred_train,
            "periods_train":periods_pred_train,
            "phases_train":phases_pred_train,
            "period_pred_train":period_pred_train,
            "n_period_pred_train":n_period_pred_train,


        }

        with open("../result/pred/res_"+self.exp_setting+".pkl","wb") as f:
            pkl.dump(data,f)
        
        # data2={"train":train_emb,"test":test_emb}
        # with open("../result/pred/res_"+self.exp_setting+"_emb.pkl","wb") as f:
        #     pkl.dump(data2,f)

        print ("================================Save results done!================================")
     

        