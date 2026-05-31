import io
import random
from typing import List
from xmlrpc.client import Boolean
from numpy import number
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.trainers
import torch
from torch import nn
#from torchtext.data import get_tokenizer
from torch import Tensor
import math
import configparser
from torch.utils.data import Dataset, DataLoader
import copy
#from torchtext.data.metrics import bleu_score
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import R2Score
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
import tslearn.metrics
import elasticnet as en
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics import mean_squared_error
import tslearn

import tokenizers
from tqdm import tqdm


#from dim_estimators import LIDL

class Utils():
    '''
    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokenizer = get_tokenizer("moses")
        result = tokenizer(text)
        result.insert(0,"<sos>")
        result.append("<eos>")
        return result

    @staticmethod
    def yield_tokens(file_path):
        with io.open(file_path, encoding = 'utf-8') as f:
            for line in f:
                yield map(str.lower, get_tokenizer("moses")(line))

    '''

    @staticmethod
    def train_tokenizer(sourcefiles: List[str]):
        tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        trainer = tokenizers.trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]","[SOS]","[EOS]"])
        tokenizer.train(sourcefiles,trainer)
        return tokenizer

    @staticmethod
    def encode_position(seq_len: int, dim_model: int, highValue = 1e4, device: torch.device = torch.device("cpu")) -> Tensor:
        pos = torch.arange(seq_len,dtype=torch.float,device=device).reshape(1,-1,1)
        dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1,1,-1)
        phase = pos / highValue ** (torch.div(dim, dim_model, rounding_mode='floor'))
        return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
    
    @staticmethod
    # this should be thoroughly optimized (perhaps by numpy)
    def norm(x1, x2):
        #return cdist(x1,x2,metric='euclidean')
        sum = 0.
        for i in range(len(x1)):
            sum += euclidean(x1[i],x2[i])
        return sum/x1.shape[0]

class ParameterProvider():
    def __init__(self, configname):
        if configname:
            self.config = configparser.ConfigParser()
            self.config.read(configname)
            self.dictionary = {
                "in_features": int(self.config['DIMENSIONS AND SIZES']['in_features']),
                "d_model": int(self.config['DIMENSIONS AND SIZES']['d_model']),
                "d_qk": int(self.config['DIMENSIONS AND SIZES']['d_qk']),
                "d_v": int(self.config['DIMENSIONS AND SIZES']['d_v']),
                "d_ff": int(self.config['DIMENSIONS AND SIZES']['d_ff']),
                "n_encoders": int(self.config['DIMENSIONS AND SIZES']['n_encoders']),
                "n_decoders": int(self.config['DIMENSIONS AND SIZES']['n_decoders']),
                "n_heads": int(self.config['DIMENSIONS AND SIZES']['n_heads']),
                "learning_rate": float(self.config['TRAINING PARAMETERS']['learning_rate']),
                "epochs": int(self.config['TRAINING PARAMETERS']['epochs']),
                "dropout": float(self.config['TRAINING PARAMETERS']['dropout'])
            }
        else:
            self.dictionary = {
                "in_features": 0,
                "d_model": 0,
                "d_qk": 0,
                "d_v": 0,
                "d_ff": 0,
                "n_encoders": 0,
                "n_decoders": 0,
                "n_heads": 0,
                "learning_rate": 1.0,
                "epochs": 0
            }
    
    def modifyWithArray(self, arr: List) -> None:
        self.dictionary = {
            "d_model": arr[0],
            "d_qk": arr[1],
            "d_v": arr[2],
            "d_ff": arr[3],
            "n_encoders": arr[4],
            "n_decoders": arr[5],
            "n_heads": arr[6],
            "epochs": self.dictionary["epochs"],
            "learning_rate": self.dictionary["learning_rate"],
            "dropout": self.dictionary["dropout"]
        }

    def getArray(self) -> List:
        return [
            self.dictionary["d_model"],
            self.dictionary["d_qk"],
            self.dictionary["d_v"],
            self.dictionary["d_ff"],
            self.dictionary["n_encoders"],
            self.dictionary["n_decoders"],
            self.dictionary["n_heads"],
            #self.dictionary["epochs"],
        ]

    def provide(self, key: str) -> any:
        return self.dictionary[key]
        
    def change(self, key: str, value: number) -> None:
        self.dictionary[key] = value

    def getChangedCopy(self,key: str, value:number):
        pp = copy.deepcopy(self)
        pp.change(key,value)
        return pp
    

class MemoryCell():
    def __init__(self):
        self.outputs = []
            # those values are filled in by the block, as they are (slightly) context-dependent
        self.dimenisonal_reductability = 0.
        self.pruningability = 0.
        self.entrophy_measure = 0.

    def add(self, value):
        self.outputs.append(value)

    #convenience and code-quality setters
    def set_dimensional_reductability(self, value):
        self.dimenisonal_reductability = value

    def set_pruningability(self, value):
        self.pruningability = value
    
    def set_entophy_measure(self, value):
        self.entrophy_measure = value

    def get_aggregated_memory(self):
        #???
        return np.concatenate(np.concatenate(self.outputs, axis = 0),axis=0)

    def clear(self):
        self.outputs = []
        self.dimenisonal_reductability = 0.
        self.pruningability = 0.
        self.entrophy_measure = 0.

    

    

class PositionalEncoding(nn.Module):

    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.d_model = config.provide('d_model')
        self.n = 10000
        self.dropout = nn.Dropout(p=config.provide('dropout'))
    

    def forward(self, input_x: Tensor) -> Tensor:

        seq_len = input_x.size(1)
        pe = torch.zeros(seq_len, self.d_model)
        for k in range(seq_len):
            for i in range(int(self.d_model/2)):
                denominator = math.pow(self.n, 2*i/self.d_model)
                pe[k, 2*i] = math.sin(k/denominator)
                pe[k, 2*i+1] = math.cos(k/denominator)
        pe = torch.stack([pe for _ in range(input_x.size(0))])
        pe = pe.cuda(device=input_x.device)
        return self.dropout(torch.add(input_x,pe))
    
class InputTransformation(nn.Module):

    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.linear = nn.Linear(config.provide("in_features"),config.provide("d_model"))
    
    def forward(self, input):
        return self.linear(input)
    

class AttentionHead(nn.Module):
    def __init__(self, config: ParameterProvider, masked: bool = False, d_v_override: int = None, d_qk_override: int = None):
        super().__init__()
        self.l = nn.Linear(100,200)
        self.masked = masked
        self.d_model = config.provide("d_model")
        self.d_qk = config.provide("d_qk") if d_qk_override is None else d_qk_override
        self.d_v = config.provide("d_v") if d_v_override is None else d_v_override
        self.WQ = nn.Linear(self.d_model,self.d_qk)
        self.WK = nn.Linear(self.d_model,self.d_qk)
        self.WV = nn.Linear(self.d_model,self.d_v)
        self.tracking = False
        self.output_memory = MemoryCell()

    def set_tracking(self, value: bool):
        self.tracking = value
    
    def clearMemory(self):
        self.output_memory.clear()

    def forward(self,input_q: Tensor, input_k: Tensor, input_v: Tensor) -> Tensor:
        Q = self.WQ(input_q)
        K = self.WK(input_k)
        V = self.WV(input_v)

        if self.masked:
            if Q.size(1) != K.size(1):
                raise TypeError('Masking can be only performed when Querry and Key Matrices have the same sizes (i.e. their product is square)')
            mask = torch.stack([torch.triu(torch.full((Q.size(1),Q.size(1)),-1*torch.inf),diagonal=1) for _ in range(Q.size(0))])
            mask = mask.to(device=Q.device)
            return_value = torch.bmm(torch.softmax(torch.add(torch.bmm(Q,torch.transpose(K,1,2)),mask)/math.sqrt(self.d_model),dim=-1),V)
            if self.tracking:
                self.output_memory.add(return_value.cpu().detach().numpy())
            return return_value
        else:
            return_value = torch.bmm(torch.softmax(torch.bmm(Q,torch.transpose(K,1,2))/math.sqrt(self.d_model),dim=-1),V)
            if self.tracking:
                self.output_memory.add(return_value.cpu().detach().numpy())
            return return_value



class MultiHeadedAttention(nn.Module):
    def __init__(self,config: ParameterProvider, masked = False, d_v_override = None, d_qk_ovveride: int = None, n_heads_override = None):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_v = config.provide("d_v") if d_v_override is None else d_v_override
        self.d_qk = config.provide("d_qk") if d_qk_ovveride is None else d_qk_ovveride
        self.n_heads = config.provide("n_heads") if n_heads_override is None else n_heads_override
        self.heads = nn.ModuleList([AttentionHead(config,masked=masked,d_qk_override=d_qk_ovveride,d_v_override=d_v_override) for _ in range(self.n_heads)])
        self.linear = nn.Linear(self.d_v*self.n_heads,self.d_model)
        self.masked = masked
        self.tracking = False
        self.output_memory = MemoryCell()

    def set_tracking(self, value: bool):
        self.tracking = value
        for h in self.heads:
            h.set_tracking(value)

    def clearMemory(self):
        self.output_memory.clear()


    def add_head_horizontal(self):
        # copy a random head:
        head_to_copy_idx = random.randint(0,len(self.heads)-1)
        self.heads.append(copy.deepcopy(self.heads[head_to_copy_idx]))
        self.n_heads += 1
        #modify the concatenation layer
        weight = self.linear.weight.detach()
        #identify position of the head
        position_head_start = head_to_copy_idx*self.d_v
        position_head_end = (head_to_copy_idx+1)*self.d_v

        random_weight_split = random.random()

        # update
        self.linear = nn.Linear(self.d_v*self.n_heads,self.d_model)

        weight[:,position_head_start:position_head_end] = weight[:,position_head_start:position_head_end] * (1. - random_weight_split)

        # bias remains unchanged (or does it?)

        weight = torch.cat((weight,weight[:,position_head_start:position_head_end]*random_weight_split),dim=-1)


        self.linear.weight = nn.Parameter(weight)


    def remove_head_horizontal(self):
        #this is a randomized, non-exact operation, so buckle up
        #determine head to be removed (currently at random, sorry)

        #think about it, like, seriously!!!!!!!

        head_to_remove_idx = random.randint(0,len(self.heads)-1)

        del self.heads[head_to_remove_idx]
        self.n_heads -= 1

        head_to_remove_idx = random.randint(0,len(self.heads)-1)

        weight = self.linear.weight

        self.linear = nn.Linear(self.d_v*self.n_heads,self.d_model)

        position_head_start = head_to_remove_idx*self.d_v
        position_head_end = (head_to_remove_idx+1)*self.d_v

        weight = torch.cat((weight[:,:position_head_start],weight[:,position_head_end:]),dim=-1)
        self.linear.weight = nn.Parameter(weight)

    def perform_memory_measures(self):
        # we count the dimensional reductability as the dimension of the output:
        cutoff = 1e-4
        threshold = 1e-2
        pca = PCA()
        pca.fit(self.output_memory.get_aggregated_memory())
        #this normalization may not be realy necessary
        eigenvalues = np.round(pca.explained_variance_ / np.max(pca.explained_variance_),3) if np.max(pca.explained_variance_) > threshold else np.round(pca.explained_variance_,3)


        cut_ratio = sum(1 for i in eigenvalues if i < cutoff) / len(eigenvalues)

        # "żyjemy w czasach, w których termostat ma bardziej stabilne poglądy niż właściciel" <3

        self.output_memory.dimenisonal_reductability = cut_ratio

        #reductability (per weights reductible in heads/Q,K,V):
        #the dimension-reductor output network probably has some characteristics of a spare matrics anyway (does it?)
        total_redundant_weights = 0
        total_weight_number = 0

        for h in self.heads:
            # pruning thresholds are decided per-matrix
            pruning_threshold = 1e-2 * np.max(h.WQ.weight.cpu().detach().numpy())
            prunable_weights = np.count_nonzero(np.absolute(h.WQ.weight.cpu().detach().numpy()) < pruning_threshold)
            total_redundant_weights += prunable_weights
            total_weight_number += np.sum(h.WQ.weight.cpu().size())

            pruning_threshold = 1e-2 * np.max(h.WV.weight.cpu().detach().numpy())
            prunable_weights = np.count_nonzero(np.absolute(h.WV.weight.cpu().detach().numpy()) < pruning_threshold)
            total_redundant_weights += prunable_weights
            total_weight_number += np.sum(h.WV.weight.cpu().size())

            pruning_threshold = 1e-2 * np.max(h.WK.weight.cpu().detach().numpy())
            prunable_weights = np.count_nonzero(np.absolute(h.WK.weight.cpu().detach().numpy()) < pruning_threshold)
            total_redundant_weights += prunable_weights
            total_weight_number += np.sum(h.WK.weight.cpu().size())

        self.output_memory.pruningability = total_redundant_weights / total_weight_number


  
    def forward(self, input_q, input_k, input_v):
        concatResult = torch.cat([h(input_q, input_k, input_v) for h in self.heads], dim = -1)
        return_value = self.linear(concatResult)
        if self.tracking:
            self.output_memory.add(return_value.cpu().detach().numpy())
        return self.linear(concatResult)


class EncoderLayer(nn.Module):
    def __init__(self,config: ParameterProvider, randomize = False):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_ff = config.provide("d_ff")
        if randomize:
            dff = self.d_ff
            ubound = int(dff * 1.1)+1
            lbound = int(dff*0.9)-1
            if lbound < 2:
                lbound = 1
            dff = random.randint(lbound,ubound)
            self.feed_forward = nn.Sequential(nn.Linear(self.d_model,dff),nn.ReLU(),nn.Linear(dff,self.d_model))
        else:
            self.feed_forward = nn.Sequential(nn.Linear(self.d_model,self.d_ff),nn.ReLU(),nn.Linear(self.d_ff,self.d_model))
        self.mha = MultiHeadedAttention(config)
        self.norm = nn.LayerNorm(self.d_model)

        self.tracking = False
        
        self.input_memory = MemoryCell()
        self.after_heads_memory = []
        self.output_memory = MemoryCell()
        
        self.after_heads_redundance = 0.
        self.output_redundance = 0.
        self.vertical_redundance = 0.


    def set_tracking(self, value: bool):
        self.tracking = value
        self.mha.set_tracking(value)

    def clearMemory(self):
        for h in self.mha.heads:
            h.clearMemory()
        self.mha.clearMemory()
        self.output_memory.clear()
        self.input_memory.clear()

    def perform_memory_measures(self):

        self.mha.perform_memory_measures()

        cutoff = 1e-4
        threshold = 1e-2
        pca = PCA()
        pca.fit(self.output_memory.get_aggregated_memory())
        #this normalization may not be realy necessary
        eigenvalues = np.round(pca.explained_variance_ / np.max(pca.explained_variance_),3) if np.max(pca.explained_variance_) > threshold else np.round(pca.explained_variance_,3)


        cut_ratio = sum(1 for i in eigenvalues if i < cutoff) / len(eigenvalues)

        # "żyjemy w czasach, w których termostat ma bardziej stabilne poglądy niż właściciel" <3

        self.output_memory.dimenisonal_reductability = cut_ratio

        #reductability (per weights reductible in the linear output):

        total_redundant_weights = 0
        total_weight_number = 0

        pruning_threshold = 1e-2 * np.max(self.feed_forward[0].weight.cpu().detach().numpy())
        total_redundant_weights += np.count_nonzero(np.absolute(self.feed_forward[0].weight.cpu().detach().numpy()) < pruning_threshold)
        total_weight_number +=  np.sum(self.feed_forward[0].weight.cpu().size())

        pruning_threshold = 1e-2 * np.max(self.feed_forward[2].weight.cpu().detach().numpy())
        total_redundant_weights += np.count_nonzero(np.absolute(self.feed_forward[2].weight.cpu().detach().numpy()) < pruning_threshold)
        total_weight_number +=  np.sum(self.feed_forward[2].weight.cpu().size())


        self.output_memory.pruningability = total_redundant_weights / total_weight_number




    def forward(self,input_data: Tensor) -> Tensor:
        if self.tracking:
            self.input_memory.add(input_data.cpu().detach().numpy())

        mha_output = self.mha(input_data,input_data,input_data)
        intermediate = self.norm(torch.add(mha_output,input_data))
        return_value = self.norm(torch.add(intermediate,self.feed_forward(intermediate)))
        if self.tracking:
            self.output_memory.add(return_value.cpu().detach().numpy())
        return return_value

class DecoderLayer(nn.Module):
    def __init__(self, config: ParameterProvider, randomize = False):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_ff = config.provide("d_ff")
        if randomize:
            dff = self.d_ff
            ubound = int(dff * 1.1)+1
            lbound = int(dff*0.9)-1
            if lbound < 2:
                lbound = 1
            dff = random.randint(lbound,ubound)
            self.feed_forward = nn.Sequential(nn.Linear(self.d_model,dff),nn.ReLU(),nn.Linear(dff,self.d_model))
        else:
            self.feed_forward = nn.Sequential(nn.Linear(self.d_model,self.d_ff),nn.ReLU(),nn.Linear(self.d_ff,self.d_model))
        self.self_mha = MultiHeadedAttention(config, masked = True)
        self.ed_mha = MultiHeadedAttention(config)
        self.norm = nn.LayerNorm(self.d_model)
        self.tracking = False

        self.input_memory = MemoryCell()
        self.after_self_mha_memory = []
        self.after_ed_mha_memory = []
        self.output_memory = MemoryCell()

        self.after_heads_self_redundance = 0.
        self.after_heads_ed_redundance = 0.
        self.output_redundance = 0.
        self.vertical_redundance = 0.

    def set_tracking(self, value: bool):
        self.tracking = value
        self.self_mha.set_tracking(value)
        self.ed_mha.set_tracking(value)

    def clearMemory(self):
        for h in self.ed_mha.heads:
            h.clearMemory()
        for h in self.self_mha.heads:
            h.clearMemory()
        self.ed_mha.clearMemory()
        self.self_mha.clearMemory()
        self.output_memory.clear()
        self.input_memory.clear()

    def perform_memory_measures(self):

        self.self_mha.perform_memory_measures()
        self.ed_mha.perform_memory_measures()

        cutoff = 1e-4
        threshold = 1e-2
        pca = PCA()
        pca.fit(self.output_memory.get_aggregated_memory())
        #this normalization may not be realy necessary
        eigenvalues = np.round(pca.explained_variance_ / np.max(pca.explained_variance_),3) if np.max(pca.explained_variance_) > threshold else np.round(pca.explained_variance_,3)


        cut_ratio = sum(1 for i in eigenvalues if i < cutoff) / len(eigenvalues)

        # "żyjemy w czasach, w których termostat ma bardziej stabilne poglądy niż właściciel" <3

        self.output_memory.dimenisonal_reductability = cut_ratio

        #reductability (per weights reductible in the linear output):

        total_redundant_weights = 0
        total_weight_number = 0

        pruning_threshold = 1e-2 * np.max(self.feed_forward[0].weight.cpu().detach().numpy())
        total_redundant_weights += np.count_nonzero(np.absolute(self.feed_forward[0].weight.cpu().detach().numpy()) < pruning_threshold)
        total_weight_number +=  np.sum(self.feed_forward[0].weight.cpu().size())

        pruning_threshold = 1e-2 * np.max(self.feed_forward[2].weight.cpu().detach().numpy())
        total_redundant_weights += np.count_nonzero(np.absolute(self.feed_forward[2].weight.cpu().detach().numpy()) < pruning_threshold)
        total_weight_number +=  np.sum(self.feed_forward[2].weight.cpu().size())


        self.output_memory.pruningability = total_redundant_weights / total_weight_number



    def forward(self,input_data: Tensor, encoder_data: Tensor) -> Tensor:
        if self.tracking:
            self.input_memory.add(input_data.cpu().detach().numpy())
        mha_output = self.self_mha(input_data,input_data,input_data)
        intermediate = self.norm(torch.add(mha_output,input_data))
        mha_output = self.ed_mha(input_data,encoder_data,encoder_data)
        intermediate = self.norm(torch.add(mha_output,intermediate))
        return_value = self.norm(torch.add(intermediate,self.feed_forward(intermediate)))
        if self.tracking:
            self.output_memory.add(return_value.cpu().detach().numpy())
        return return_value
    

class EncoderStack(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.n_encoders = config.provide("n_encoders")
        self.encoders = nn.ModuleList([EncoderLayer(config) for _ in range(self.n_encoders)])
        self.tracking = False
        self.config = config

    def set_tracking(self,value: bool):
        self.tracking = value
        for e in self.encoders:
            e.set_tracking(value)

    def clearMemory(self):
        for e in self.encoders:
            e.clearMemory()
    
    def add_encoder(self, location: int = -1):
        #naive method:
        self.encoders.insert(location,[EncoderLayer(self.config)])

    
    def forward(self, input_data: Tensor) -> Tensor:
        for l in self.encoders:
            input_data = l(input_data)
        return input_data

class NumericalOut(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.linear = nn.Linear(config.provide("d_model"),config.provide("in_features"))
    
    def forward(self, input_data: Tensor) -> Tensor:
        #return torch.softmax(self.linear(input_data), dim = -1)
        return self.linear(input_data)


class DecoderStack(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.n_decoders = config.provide("n_decoders")
        self.decoders = nn.ModuleList([DecoderLayer(config) for _ in range(self.n_decoders)])
        self.tracking = False
        self.config = config

    def set_tracking(self,value: bool):
        self.tracking = value
        for d in self.decoders:
            d.set_tracking(value)

    def clearMemory(self):
        for d in self.decoders:
            d.clearMemory()

    def add_decoder(self, location: int = -1):
        #naive method:
        self.decoders.insert(location,[DecoderLayer(self.config)])

    def forward(self, input_data: Tensor, ed_data: Tensor) -> Tensor:
        for l in self.decoders:
            input_data = l(input_data,ed_data)
        return input_data

class Router:
    def __init__(self, idx_from, idx_left, idx_right, config: ParameterProvider):
        self.idx_from = idx_from
        self.idx_left = idx_left
        self.idx_right = idx_right



class CustomDataSet(Dataset):
    def __init__(self, infile: str, window_length = 32, prediction_window = 7, require_date_split = True, drop_idx_column = False):

        file = open(infile,'r')
        df = pd.read_csv(file)
        df = df.dropna()
        if require_date_split:
            df[["day", "month", "year"]] = df["Date"].str.split("/", expand = True)
            df['day'] = df['day'].astype(float)
            df['month'] = df['month'].astype(float)
            df['year'] = df['year'].astype(float)
            df = df.drop(['Date'],axis=1)
        if drop_idx_column:
            df = df.drop(['ID'],axis=1)
        #df = df.apply(pd.to_numeric)
        data = torch.tensor(df.values)

        print(list(df.columns))

        # normalize (is this correct way to normalize this?)
        
        for i in range(data.size(1)):
            epsilon = torch.zeros_like(data[:,i])
            epsilon.fill_(1e-6)
            data[:,i] = (data[:,i] - torch.min(data[:,i])) / torch.max(epsilon,(torch.max(data[:,i]) - torch.min(data[:,i])))
        
        #data = torch.nn.functional.normalize(data, dim=0)

        source_samples = []
        shifted_target = []
        target_samples = []

        for i in range(window_length, data.size()[0]-prediction_window):
            source_samples.append(data[i-window_length:i,:])
            shifted_target.append(data[i-1:i+prediction_window-1,:])
            target_samples.append(data[i:i+prediction_window,:])

        self.L = len(range(window_length, data.size()[0]-prediction_window))

        self.Xe = source_samples
        self.Xd = shifted_target
        self.Y = target_samples
        '''
                self.Xe = torch.nn.utils.rnn.pad_sequence(self.Xe,batch_first=True)
                self.Xd = torch.nn.utils.rnn.pad_sequence(self.Xd,batch_first=True)
                self.Y = torch.nn.utils.rnn.pad_sequence(self.Y,batch_first=True)
        '''
    def __len__(self):
        return self.L

    def __getitem__(self, index):
        _xe = self.Xe[index]
        _xd = self.Xd[index]
        _y = self.Y[index]
        return _xe.to(torch.float), _xd.to(torch.float), _y.to(torch.float)

    def getSets(self, split = 0.8):
        train_size = int(0.8 * len(self))
        test_size = len(self) - train_size

        # Created using indices from 0 to train_size.
        train_dataset = torch.utils.data.Subset(self, range(train_size))

        # Created using indices from train_size to train_size + test_size.
        test_dataset = torch.utils.data.Subset(self, range(train_size, train_size + test_size))

        return train_dataset,test_dataset
    
class InputTransformation(nn.Module):

    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.linear = nn.Linear(config.provide("in_features"),config.provide("d_model"))

    def forward(self, input):
        return self.linear(input)


class BudFormer(nn.Module):
    def __init__(self, linkage_matrix: np.array, cluster_centers, config: ParameterProvider):
        super().__init__()
        self.config = config
        self.n_buds = 0
        self.buds = {}
        self.decoder_buds = nn.ModuleDict() #identified by indices
        self.encoder_buds = nn.ModuleDict()
        self.leaf_outs = nn.ModuleDict()
        self.cluster_paths = {} #also identified by indices
        self.linkage_matrix = linkage_matrix
        self.cluster_centers = cluster_centers
        self.n = self.linkage_matrix.shape[0]+1
        self.encoder_input_transformation = InputTransformation(config)
        self.decoder_input_transformation = InputTransformation(config)
        self.pos_encoding_in = PositionalEncoding(config)
        self.pos_encoding_out = PositionalEncoding(config)

    def form(self, n_clusters):

        bud_max = 2*self.n-2
        self.encoder_buds[str(int(bud_max))] = EncoderStack(self.config)
        self.decoder_buds[str(int(bud_max))] = DecoderStack(self.config)

        self.cluster_paths[bud_max] = [bud_max]

        self.leaves = [bud_max]
        
        for i in range(n_clusters):
            leaf_id = bud_max - i
            leaf_left = self.linkage_matrix[-i-1][0]
            leaf_right = self.linkage_matrix[-i-1][1]

            self.encoder_buds[str(int(leaf_left))] = EncoderStack(self.config)
            self.decoder_buds[str(int(leaf_left))] = DecoderStack(self.config)
            self.encoder_buds[str(int(leaf_right))] = EncoderStack(self.config)
            self.decoder_buds[str(int(leaf_right))] = DecoderStack(self.config)

            self.leaves.remove(leaf_id)
            self.leaves.append(int(leaf_left))
            self.leaves.append(int(leaf_right))

            self.cluster_paths[leaf_left] = self.cluster_paths[leaf_id] + [int(leaf_left)]
            self.cluster_paths[leaf_right] = self.cluster_paths[leaf_id] + [int(leaf_right)]
        
        for l in self.leaves:
            self.leaf_outs[str(int(l))] = NumericalOut(self.config)
            
            

    def forward(self, encoder_input: Tensor, decoder_input: Tensor, cluster: int):

        #check for the closest cluster


        path = self.cluster_paths[cluster]

        encoder_x = self.encoder_input_transformation(encoder_input)
        encoder_x = self.pos_encoding_in(encoder_x)
        decoder_x = self.decoder_input_transformation(decoder_input)
        decoder_x = self.pos_encoding_out(decoder_x)

        #propagade encoder
        for p in path:
            encoder_x = self.encoder_buds[str(p)](encoder_x)
        
        #propagate decoder
        for p in path:
            decoder_x = self.decoder_buds[str(p)](decoder_x,encoder_x)


        output = self.leaf_outs[str(path[-1])](decoder_x)
        return output
    

    def split_into_batches(self, xe, xd, y):

        
        buckets_xe = []
        buckets_xd = []
        buckets_y = []

        for i in range(self.n*2-1):
            buckets_xe.append([])
            buckets_xd.append([])
            buckets_y.append([])


        size = xe.shape[0]

        ctr = 0 # debug

        for i in range(size):
            min_distance = np.inf
            min_idx = self.leaves[0]
            for j, c in enumerate(self.cluster_centers):
                if j in self.leaves:
                    distance = tslearn.metrics.dtw(xe[i].detach().numpy(),c)
                    ctr += 1
                    if distance < min_distance:
                        min_idx = j
                        min_distance = distance
            
            buckets_xe[min_idx].append(xe[i])
            buckets_xd[min_idx].append(xd[i])
            buckets_y[min_idx].append(y[i])
        
        #print(ctr) #debug
        
        buckets_pruned = {}     

        for i in range(self.n*2-1):
            if len(buckets_xe[i]) > 0:
                buckets_pruned[i] = []
                buckets_pruned[i].append(torch.stack(buckets_xe[i]))
                buckets_pruned[i].append(torch.stack(buckets_xd[i]))
                buckets_pruned[i].append(torch.stack(buckets_y[i]))                

        return buckets_pruned



    def auto_output(self,xe,xd,device):

        output = []

        size = xe.shape[0]

        for i in range(size):
            min_distance = np.inf
            min_idx = self.leaves[0]
            for j, c in enumerate(self.cluster_centers):
                if j in self.leaves:
                    distance = tslearn.metrics.dtw(xe[i].detach().numpy(),np.stack(c))
                    if distance < min_distance:
                        min_idx = j
                        min_distance = distance
            output.append(self(torch.unsqueeze(xe[i].cuda(device),0),torch.unsqueeze(xd[i].cuda(device),0),min_idx))
        
        return torch.stack(output)




    def train_cuda(self, train_dataset: CustomDataSet, device: int, batch_size = 32, lr: float = 0.001, epochs: int = 1, verbose_delay = 100, optimizer = None) -> None:
    
        self.cuda(device=device)
        criterion = nn.MSELoss().cuda(device=device)

        self.train()
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters(),lr=lr)

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        epoch_loss_history = []

        epoch_loss = 0.
        for epoch in range(epochs):

            print('Epoch ' + str(epoch)+' of '+str(epochs))
            epoch_loss = 0.
            for i, (xe, xd, y) in tqdm(enumerate(data_loader)):


                if verbose_delay > 0 and i%verbose_delay== 0:
                    print(str(i) + " of " + str(len(data_loader)))

                data_buckets = self.split_into_batches(xe,xd,y)

                for k, v in data_buckets.items():
                    
                    xe = v[0]
                    xd = v[1]
                    y = v[2]

                    last_loss = 0.
                    xe = xe.cuda(device)
                    xd = xd.cuda(device)

                    y = y.cuda(device)

                    optimizer.zero_grad()
                    output = self(xe, xd, int(k))

                    loss = criterion(output,y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()

                    last_loss = loss.item()
                    epoch_loss += float(last_loss)
                    del loss
                    del output
            epoch_loss /= len(data_loader)

            epoch_loss_history.append(epoch_loss)

            print("Epoch loss: "+str(epoch_loss))
        
        return epoch_loss, epoch_loss_history
    
    def validate_cuda(self, test_dataset: CustomDataSet, device: int, batch_size = 32, verbose_delay = 100):
        self.cuda(device=device)
        criterion = nn.MSELoss().cuda(device=device)
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        final_mse = 0.
        with torch.no_grad():
            for i, (xe, xd, y) in enumerate(data_loader):

                data_buckets = self.split_into_batches(xe,xd,y)

                for k, v in data_buckets.items():
                    
                    xe = v[0]
                    xd = v[1]
                    y = v[2]

                    xe = xe.cuda(device)
                    xd = xd.cuda(device)

                    y = y.cuda(device)
                    final_mse += criterion(self(xe,xd,int(k)),y)
        
        return  final_mse / len(data_loader)

    def output_and_show(self, input: Tensor, output: Tensor, device: int, visualize_index: int = 0):
        decoder_input = torch.cat((torch.unsqueeze(input[0],dim=0),output[0:-1]))
        
        self.cuda(device=device)
        predicted_output = self.auto_output(torch.unsqueeze(input,dim=0),torch.unsqueeze(decoder_input,dim=0),device).cpu()
        input_list = input[:,visualize_index].tolist()
        output_list = output.cpu()[:,visualize_index].tolist()
        predicted_list = predicted_output[0,0].detach()[:,visualize_index].tolist()
        numbers_pre_prediction = list(range(len(input_list)))
        numbers_post_prediction = list(range(len(input_list),len(input_list)+len(output_list)))
        print("R^2", R2Score()(torch.tensor(output_list),torch.tensor(predicted_list)), flush = True)
        plt.plot(numbers_pre_prediction, input_list,'k-')
        plt.plot(numbers_post_prediction, output_list,'b-')
        plt.plot(numbers_post_prediction, predicted_list,'r-')
        plt.show()

    def output_and_show_only_prediction(self, input: Tensor, output: Tensor, device: int, visualize_index: int = 0):
        decoder_input = torch.cat((torch.unsqueeze(input[0],dim=0),output[0:-1]))
        
        self.cuda(device=device)
        predicted_output = self(torch.unsqueeze(input,dim=0).to(device),torch.unsqueeze(decoder_input,dim=0).to(device)).cpu()
        input_list = input[:,visualize_index].tolist()
        output_list = output.cpu()[:,visualize_index].tolist()
        predicted_list = predicted_output[0].detach()[:,visualize_index].tolist()
        numbers_post_prediction = list(range(len(input_list),len(input_list)+len(output_list)))
        print("R^2", R2Score()(torch.tensor(output_list),torch.tensor(predicted_list)), flush = True)
        print("MSE:",mean_squared_error(torch.tensor(output_list),torch.tensor(predicted_list)))
        plt.title('Predicted vs. true result')
        plt.plot(output_list,'b-',label='True data')
        plt.plot(predicted_list,'r-',label = 'Predicted data')
        plt.legend()
        plt.show()

    def output_and_save(self, input: Tensor, output: Tensor, device: int, visualize_index: int = 0, location = ".", name = "fig"):
            
        decoder_input = torch.cat((torch.unsqueeze(input[0],dim=0),output[0:-1]))
        self.cuda(device=device)
        predicted_output = self.auto_output(torch.unsqueeze(input,dim=0),torch.unsqueeze(decoder_input,dim=0),device).cpu()
        input_list = input[:,visualize_index].tolist()
        output_list = output.cpu()[:,visualize_index].tolist()
        predicted_list = predicted_output[0,0].detach()[:,visualize_index].tolist()
        numbers_pre_prediction = list(range(len(input_list)))
        numbers_post_prediction = list(range(len(input_list),len(input_list)+len(output_list)))
        print("R^2", R2Score()(torch.tensor(output_list),torch.tensor(predicted_list)), flush = True)
        plt.plot(numbers_pre_prediction, input_list,'k-')

        plt.plot(numbers_post_prediction, output_list,'b-')
        plt.plot(numbers_post_prediction, predicted_list,'r-')
        plt.savefig(location + '/' + name + 'fig.png')

    def return_numbers(self, input: Tensor, output: Tensor, device: int, visualize_index: int = 0, location = ".", name = "fig"):
        decoder_input = torch.cat((torch.unsqueeze(input[0],dim=0),output[0:-1]))  
        self.cuda(device=device)
        predicted_output = self.auto_output(torch.unsqueeze(input,dim=0),torch.unsqueeze(decoder_input,dim=0),device).cpu()
        input_list = input[:,visualize_index].tolist()
        output_list = output.cpu()[:,visualize_index].tolist()
        predicted_list = predicted_output[0,0].detach()[:,visualize_index].tolist()
        return input_list, output_list, predicted_list


    def output_and_save_only_prediction(self, input: Tensor, output: Tensor, device: int, visualize_index: int = 0, location = ".", name = "fig"):
        decoder_input = torch.cat((torch.unsqueeze(input[0],dim=0),output[0:-1]))
        
        self.cuda(device=device)
        predicted_output = self.auto_output(torch.unsqueeze(input,dim=0),torch.unsqueeze(decoder_input,dim=0),device).cpu()
        input_list = input[:,visualize_index].tolist()
        output_list = output.cpu()[:,visualize_index].tolist()
        predicted_list = predicted_output[0,0].detach()[:,visualize_index].tolist()
        numbers_pre_prediction = list(range(len(input_list)))
        numbers_post_prediction = list(range(len(input_list),len(input_list)+len(output_list)))
        print("R^2", R2Score()(torch.tensor(output_list),torch.tensor(predicted_list)), flush = True)
        print("MSE",mean_squared_error(torch.tensor(output_list),torch.tensor(predicted_list)),flush=True)
        plt.plot(numbers_pre_prediction, input_list,'k-')
        plt.plot(numbers_post_prediction, output_list,'b-')
        plt.plot(numbers_post_prediction, predicted_list,'r-')
        plt.show()
        plt.title('Predicted vs. true result')
        plt.plot(output_list,'b-',label='True data')
        plt.savefig(location + '/' + name + '_output.png')
        plt.plot(predicted_list,'r-',label = 'Predicted data')
        plt.savefig(location + '/' + name + 'prediction.png')
        plt.legend()
        plt.show()
        return "R^2 " + str(R2Score()(torch.tensor(output_list),torch.tensor(predicted_list))) + "; MSE " + str(mean_squared_error(torch.tensor(output_list),torch.tensor(predicted_list)))


    def long_range_output_and_show(self, input: Tensor, output: Tensor, device: int, visualize_index: int = 0):
        decoder_input = torch.cat((torch.unsqueeze(input[0],dim=0),output[0:-1]))
        
        self.cuda(device=device)
        predicted_output = self(torch.unsqueeze(input,dim=0).to(device),torch.unsqueeze(decoder_input,dim=0).to(device)).cpu()
        print(output.size())
        print(predicted_output.size())
        plt.plot(output.cpu()[:,visualize_index],'b-')
        plt.plot(predicted_output.detach()[0,:,visualize_index],'r-')
        plt.show()

        
    def long_range_output_and_save(self, input: Tensor, output: Tensor, device: int, visualize_index: int = 0, location = ".", name = "fig"):
        decoder_input = torch.cat((torch.unsqueeze(input[0],dim=0),output[0:-1]))
        
        self.cuda(device=device)
        predicted_output = self(torch.unsqueeze(input,dim=0).to(device),torch.unsqueeze(decoder_input,dim=0).to(device)).cpu()

        plt.plot(output.cpu()[:,visualize_index],'b-')
        plt.savefig(location + '/' + name + '_long_range_output.png')
        plt.plot(predicted_output.detach()[0,:,visualize_index],'r-')
        plt.savefig(location + '/' + name + '_long_range_prediction.png')
        plt.show()

        return ""