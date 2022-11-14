import torch
import torch.nn as nn

from layers.embed import DataEmbedding
from layers.attention import FullAttention,AttentionLayer
from layers.encoder import Encoder, EncoderLayer, ConvLayer
from layers.decoder import nonPeriodic,Periodic,MLP


class DeepFS(nn.Module):
    def __init__(self,args,embed_type="fixed"):
        super(DeepFS,self).__init__()

        self.args = args
        self.enc_in = args.enc_in
        self.freq = args.freq

        
        self.input_len = args.input_len
        self.out_len = args.out_len
        self.embed_size = args.embed_size
        self.d_ff = args.embed_size

        self.n_heads = args.n_heads
        self.e_layers = args.e_layers
        self.distil = args.distil

        self.base = args.base

        self.dropout = args.dropout
        self.activation = args.activation

        self.output_attention = args.output_attention
        

        self.enc_embedding = DataEmbedding(self.enc_in, self.embed_size, embed_type, self.freq, self.dropout)


        # Attention
        Attn = FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, attention_dropout=self.dropout, output_attention=self.output_attention), 
                                self.embed_size, self.n_heads, mix=False),
                    self.embed_size,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.embed_size
                ) for l in range(self.e_layers-1)
            ] if self.distil else None,
            norm_layer=nn.LayerNorm(self.embed_size)
        )

        self.nn2 = MLP(self.input_len,args.embed_size,args.embed_size)
        self.period_projection = Periodic(args)
        self.non_period_projection = nonPeriodic(args)



    def forward(self,timestamps,x_enc,x_mark_enc=None,enc_self_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc,self.args.is_sim)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        temp = enc_out
        period_pred,weights_pred,periods_pred,phases_pred = self.period_projection(enc_out,timestamps)
        enc_out = torch.mean(enc_out,axis=2)
        nperiod_pred = self.non_period_projection(enc_out)
        pred = self.args.alpha*period_pred+self.args.beta*nperiod_pred
        
        return pred,weights_pred,periods_pred,phases_pred,period_pred,nperiod_pred,temp

        
        
        
        
        

    

        