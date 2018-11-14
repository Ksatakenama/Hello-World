# coding: UTF-8
#2018/05/08
#by Jo Aoe
#optimize緩やかに
#MNetをENVNetv2元に
#EPvsSCvsHV判定

import os
import numpy as np
import argparse
import glob
import wave
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import cPickle as pickle
import datetime
import itertools
from chainer import serializers

import scipy.io
import h5py
from sklearn import utils
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
import random
import gc
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import math
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
from multiprocessing import Process
from sklearn.metrics import accuracy_score
import pickle

plt.ioff()

# 中間結果等を保存するディレクトリ
save_dir = '/work/project/MEG_DiagnosisFromRawSignals/SpikeDetection/Epi_and_other_disease/log'

# 保存先ディレクトリ名を生成
# フラグを設定して付けるのが大変なので，ディレクトリ名は日付から生成する
now = datetime.datetime.today()
dir_name = '{0:%Y-%m-%d_%H-%M-%S_log_mnetv2_EPvsSCvsHV}'.format(now)
save_dir = os.path.join(save_dir, dir_name)
# 保存先ディレクトリを生成
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
    os.chmod(save_dir,0777)

parser = argparse.ArgumentParser(description='Chainer')

# GPU ID
parser.add_argument('--gpu', default=0, type=int,help='GPU ID (negative value indicates CPU)')
# 最大エポック数
parser.add_argument('--max_epochs', default=10, help='max epochs', type=int)
# Batch size
parser.add_argument('--batch_size', default=2, help='size of batch', type=int)
# 1人の被検者あたりに何個のデータを取るか
parser.add_argument('--num_for_each', default=3, help='number for each patient', type=int)
# weight_decayパラメータ
parser.add_argument('--weight_decay', default=0.0005, help='weight decay', type=float)
# 学習率パラメータ
parser.add_argument('--lr', default=0.001, help='learning rate', type=float)
# fold number for K-fold
parser.add_argument('--fold_number', default=2, help='fold number', type=int)

# コマンドラインの引数を解釈
args = parser.parse_args()
# さらに，辞書形式に変換
args = vars(args)
GPU_ID = args['gpu']
EPOCH_N = args['max_epochs']
BATCH_SIZE = args['batch_size']
NUM_FOR_EACH = args['num_for_each']
WEIGHT_DECAY = args['weight_decay']
LR = args['lr']
FOLD_N= args['fold_number']

if GPU_ID >= 0:
    cuda.check_cuda_available()

fpath = '/work/project/MEG_DiagnosisFromRawSignals/SpikeDetection/Epi_and_other_disease/data_info.csv'

df = pd.read_csv(fpath)

#めんどくさいので除去するもののフラグを付けて除去
#find無ければ-1
Jokyo_flag=np.zeros((1,(df.shape)[0]))
for i in xrange((df.shape)[0]):
    if (df.path[i].find('trig') != -1 or df.path[i].find('alpha') != -1 or df.path[i].find('beta') != -1 or df.path[i].find('delta') != -1 or df.path[i].find('hgamma1') != -1 or df.path[i].find('lgamma1') != -1 or df.path[i].find('theta') != -1 or df.path[i].find('view') != -1 or df.path[i].find('SC0016_20160809') != -1 or df.path[i].find('SC0008_20170119') != -1 or df.path[i].find('NV0061_20140108') != -1 or df.path[i].find('PD0023_20150827') != -1 or df.path[i].find('PD0026_20150602') != -1 or df.path[i].find('ET0020_20150520') != -1 or df.path[i].find('E0026_20160322') != -1 or df.path[i].find('E0001_20150203') != -1 or df.path[i].find('E0007_20130313') != -1 or df.path[i].find('E0025_20150416') != -1 or df.path[i].find('MEG0058_20150324') != -1 or df.path[i].find('E0011_20141118') != -1 or df.path[i].find('EP0037_20151210') != -1 or df.path[i].find('E0010_20150410') != -1 or df.path[i].find('EP0036_20160114') != -1 or df.path[i].find('EP0057_20150514') != -1 or df.path[i].find('MEG0046_20150519') != -1 or df.path[i].find('EP0065_20150217') != -1 or df.path[i].find('SC0020_20150910') != -1 or df.path[i].find('SC0021_20150825') != -1 or df.path[i].find('SC0022_20150422') != -1 or df.path[i].find('SC0026_20141121') != -1 or df.path[i].find('SC0030_20140902') != -1 or df.path[i].find('SC0031_20140724') != -1 or df.path[i].find('SC0032_20131031') != -1 or df.path[i].find('SC0033_20131028') != -1 or df.path[i].find('SC0034_20130806') != -1):
        Jokyo_flag[0,i]=1
        #除去するやつら　SC16:noise多い 他はデータの怪しさを理由に除去,SC8:謎の２回というコメント,NV61:data長なぜか400s,PD23:specifiedsamplecountの項目有。短い,PD26:名前2kなのに1k,ET20,名前2kなのに1k,E26:downsampleなぜかうまくできてない,E7:診断怪しい,E7:重複,E25:重複,MEG58:重複
    if (df.disease[i]=='PA' or df.disease[i]=='ST' or df.path[i]=='/work/project/MEG_DiagnosisFromRawSignals/data/Data_Yanagisawa/PD0027_20150515/_spont__02.mat' or df.path[i]=='/work/project/MEG_DiagnosisFromRawSignals/data/Data_Yanagisawa/PD0028_20150513/_spont__02.mat' or df.path[i]== '/work/project/MEG_DiagnosisFromRawSignals/data/Data_Yanagisawa/PD0029_20150324/_spont__02.mat' or df.path[i]=='/work/project/MEG_DiagnosisFromRawSignals/data/Data_Yanagisawa/ET0021_20150320/_spont__02.mat' or df.path[i]=='/work/project/MEG_DiagnosisFromRawSignals/data/Data_Yanagisawa/SC0011_20161117/_spont__02.mat'):
        #複数ファイルにまたがるものも除去する
        Jokyo_flag[0,i]=1

df2 = pd.DataFrame(Jokyo_flag.T, columns=['Jokyo_flag'])
df3 = df.join(df2)

data_use = df3[df3.Jokyo_flag != 1]

#keyの抜けてるとこあるのを直す
data_use = data_use.reset_index(drop=True)

#所見番号の取得

import os
dict, file = os.path.split(data_use.path[0])

dirc=[]

for i in xrange(data_use.shape[0]):
    dict, file = os.path.split(data_use.path[i])
    dirc.append(dict)

a=np.array(dirc)
u, indices = np.unique(a, return_index=True)

shoken_id=np.zeros((data_use.shape[0],1))
for i in xrange(indices.shape[0]):
    if(i<indices.shape[0]-1):
        shoken_id[indices[i]:indices[i+1],0]=i
    else:
        shoken_id[indices[i]:, 0] = i

df4 = pd.DataFrame(shoken_id, columns=['Shoken_id']);
data_use = data_use.join(df4)

id=int(data_use['Shoken_id'][-1:].values[0])+1

#Epiのファイルは初めのファイルのみ使う。基本*1.matだが、*2.matしかないもの及び、*1.matのデータ長が短いものがあるので、その場合*2.matを用いる。
Jokyo_flag_2 = np.zeros((1,(data_use.shape)[0]))
for i in xrange(id):
    #所見id=iのデータフレーム
    data_tmp = data_use[data_use.Shoken_id==i]
    #てんかんであれば最初のファイルを用いる処理を適応する
    if(data_tmp.disease[data_tmp.index[0]]=='EP'):
        #初めのファイルが1.matであることを確認する
        if(data_tmp.path[data_tmp.index[0]].find('1.mat')!=-1):
            #1.mat fileが200s以上か確認する
            if(data_tmp.sample_num[data_tmp.index[0]] > 200000):
                #1.mat fileのみ残し他のファイルはデータフレームから除去するフラグを入れる
                Jokyo_flag_2[0,data_tmp.index[1:]]=1
            else:
                Jokyo_flag_2[0, data_tmp.index[0]] = 1
                Jokyo_flag_2[0, data_tmp.index[2:]] = 1
        else:
            #初めのファイルが2.matの場合
            Jokyo_flag_2[0, data_tmp.index[1:]] = 1

df_tmp = pd.DataFrame(Jokyo_flag_2.T, columns=['Jokyo_flag_2'])
data_use_2 = data_use.join(df_tmp)
data_use_3 = data_use_2[data_use_2.Jokyo_flag_2 != 1]

# keyの抜けてるとこあるのを直す
data_use = data_use_3.reset_index(drop=True)

##いらんものを除去

Jokyo_flag_3 = np.zeros((1,(data_use.shape)[0]))
for i in xrange(id):
    #所見id=iのデータフレーム
    data_tmp = data_use[data_use.Shoken_id==i]
    if(data_tmp.disease[data_tmp.index[0]]=='ET' or data_tmp.disease[data_tmp.index[0]]=='PD'):
        Jokyo_flag_3[0,data_tmp.index[:]]=1

df_tmp_ = pd.DataFrame(Jokyo_flag_3.T, columns=['Jokyo_flag_3'])
data_use_2_ = data_use.join(df_tmp_)
data_use_3_ = data_use_2_[data_use_2_.Jokyo_flag_3 != 1]
# keyの抜けてるとこあるのを直す
data_use = data_use_3_.reset_index(drop=True)
#所見番号を直す
s_c=0
for i in xrange(data_use.index.shape[0]):
    #最後は次が無いので
    if(i==data_use.index.shape[0]-1):
        data_use.Shoken_id[i] = s_c
    else:
        #次の所見番号が同じなら所見番号増やさない
        if(data_use.Shoken_id[i]==data_use.Shoken_id[i+1]):
            data_use.Shoken_id[i] = s_c
        else:
            data_use.Shoken_id[i] = s_c
            s_c += 1
#所見の数の更新
id=int(data_use['Shoken_id'][-1:].values[0])+1
'''
data_use.to_csv();
data_use.to_csv('/work/project/MEG_DiagnosisFromRawSignals/SpikeDetection/Epi_and_other_disease/MNetv2_3disease.csv')
'''

#所見ごとのsample numberをnumpy arrayとして作る
sample_num=np.zeros((id,id))
for j in xrange(id):
    #print j
    rn = 0  # repeat number
    for i in xrange(data_use['Shoken_id'].size):
        #print i
        if((data_use['Shoken_id'])[i]==j):
            #print rn
            sample_num[j,rn] = (data_use['sample_num'])[i]
            rn = rn+1


#所見ごとの病気ラベルををnumpy arrayとして作る
#NV:0,EP:1,SC:2
disease_flag=np.zeros((id,id))
for j in xrange(id):
    #print j
    rn = 0  # repeat number
    for i in xrange(data_use['Shoken_id'].size):
        #print i
        if((data_use['Shoken_id'])[i]==j):
            #print rn
            if(data_use.disease[i]=='NV'):
                disease_flag[j, rn] = 0
                rn = rn + 1
            elif(data_use.disease[i]=='EP'):
                disease_flag[j, rn] = 1
                rn = rn + 1
            else:
                disease_flag[j, rn] = 2
                rn = rn + 1

#所見ごとのpathをnumpy arrayとして作る
file_path=np.full((id,id),'',dtype=object)
for j in xrange(id):
    #print j
    rn = 0  # repeat number
    for i in xrange(data_use['Shoken_id'].size):
        #print i
        if((data_use['Shoken_id'])[i]==j):
            #print rn
            file_path[j,rn] = (data_use['path'])[i]
            rn = rn+1

#file_path[0,:]

#所見ごとのサンプル数合計
#初めと終わりの10s除去
tmp1=(np.sum(sample_num,axis=1)-20000)[:,np.newaxis]
#乱数のシードを固定しておく
np.random.seed(1989)

def balanced_accuracy_score(y_true, y_pred):
    C = metrics.confusion_matrix(y_true,y_pred).astype(np.float32)
    # print 'balanced with {0:d} labels!'.format(C.shape[0])
    return np.mean(np.diag(C) / np.sum(C,axis=1))

def Load_Batch(I, NUM_FOR_EACH,Rand_id, index,Batch_size, File_path, Sample_num, Total_patient,whole_data):
    result=np.zeros((Batch_size,1,160,800),dtype=np.float32) #800sample,160chを仮定
    label=[]
    for j in xrange(Batch_size):
        if (I + j >= Rand_id.shape[0]):
            # 最後のバッチの大きさが足りない場合
            pass
        else:
            Patient_N = index[Rand_id[I + j] / NUM_FOR_EACH]
            #同じ所見だとしても毎回異なる場所からデータを取得する
            s_n = int(tmp1[Patient_N,0]*np.random.rand())
            tmp = whole_data[Patient_N][s_n + 10000:s_n + 10800, :]
            tmp = tmp.transpose()
            tmp = tmp[np.newaxis, np.newaxis, :, :]
            tmp = tmp.astype(np.float32)
            result[j, :, :, :] = tmp
            label.append(disease_flag[Patient_N, 0])

    label=np.array(label)

    #channel毎のscale
    for a in range(result.shape[0]):
        for b in range(result.shape[2]):
            result[a, 0, b, :] = preprocessing.scale(result[a, 0, b, :] )

    result = result[0:label.shape[0], :, :, :]

    #print 'debug'
    #save_debug = os.path.join(save_dir, 'debug.npz')
    #np.savez(save_debug, result=result, label=label)
    #print result.shape
    return result,label

def Load_Test_Batch(Index_num,r_n,ama, File_path, Disease_flag,whole_data):
    result=np.zeros((ama,1,160,800),dtype=np.float32)
    label = np.ones((ama),dtype=np.int32)*Disease_flag[Index_num,0]

    for j in xrange(ama):
        tmp = whole_data[Index_num][r_n*10+j*800+10000:r_n*10+(j+1)*800+10000, :]
        tmp = tmp.transpose()
        tmp = tmp[np.newaxis, np.newaxis, :, :]
        tmp = tmp.astype(np.float32)
        result[j, :, :, :] = tmp

    #channel毎のscale
    for a in range(result.shape[0]):
        for b in range(result.shape[2]):
            result[a, 0, b, :] = preprocessing.scale(result[a, 0, b, :] )

    return result,label

def forward(x_data, y_data, model_use, train=True ):
    x, t = Variable(x_data), Variable(y_data)
    test = not train

    h = F.relu(model_use.conv1(x))
    h = F.relu(model_use.conv2(h))
    h = F.max_pooling_2d(h, (1, 2), stride=(1,2))
    h = F.reshape(h, (h.data.shape[0], h.data.shape[2], h.data.shape[1], h.data.shape[3]))

    h = F.relu(model_use.conv3(h))
    h = F.relu(model_use.conv4(h))
    h = F.max_pooling_2d(h, (5,3))
    h = F.relu(model_use.conv5(h))
    h = F.relu(model_use.conv6(h))
    h = F.max_pooling_2d(h, (1, 2))
    h = F.relu(model_use.conv7(h))
    h = F.relu(model_use.conv8(h))
    h = F.max_pooling_2d(h, (1, 2))
    h = F.relu(model_use.conv9(h))
    h = F.relu(model_use.conv10(h))
    h = F.max_pooling_2d(h, (1, 2))

    h = F.relu(model_use.norm1(model_use.fc11(h), test=test))
    h = F.dropout(h, train=train)
    h = F.relu(model_use.norm2(model_use.fc12(h), test=test))
    h = F.dropout(h, train=train)
    y = model_use.fc13(h)

    if train:
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t), F.softmax(y)
    else:
        return F.softmax(y)

# 計算コンディションを出力する
# まずは，プログラムの引数
log_txt=''
log_txt += '-----------Arguments-----------\n'
for key, value in args.iteritems():
    log_txt += key + ': ' +  value.__str__() + '\n'
log_txt += '\n'

#Cross validation K-fold
#FOLD_N=10
skf = StratifiedKFold(disease_flag[:,0], n_folds=FOLD_N, shuffle=True)
# Currnet fold number
c_f=0

#データ数
#EPが一番多くて140所見なので、拡張性無いけどとりあえず
N_data=140*NUM_FOR_EACH*3
#*9をしているのは、ラベルに用いていない2をはじめに配列に入れておくことで、後で、除去できるようにするため。
#train dataにおいて、各fold,各epochでのラベルの予測を保存する配列
train_labels_pred=np.ones((FOLD_N,N_data,EPOCH_N))*9
#train dataにおいて、各fold,各epochでのラベルの真値を保存する配列
train_labels_true=np.ones((FOLD_N,N_data,EPOCH_N))*9

#test data数の計算
N_test_data=0
for i in xrange(id):
    N_test_data += tmp1[i,0]/800
N_test_data = int(N_test_data)
#test dataにおいて、各fold,各epochでのラベルの予測を保存する配列
test_labels_pred=np.ones((FOLD_N,N_test_data,EPOCH_N))*9
#test dataにおいて、各fold,各epochでのラベルの真値を保存する配列
test_labels_true=np.ones((FOLD_N,N_test_data,EPOCH_N))*9

#所見accを保存する配列
shoken_acc_matrix=np.zeros((FOLD_N,EPOCH_N))
#感度、特異度を保存する配列
Sensitivity_matrix=np.zeros((FOLD_N,EPOCH_N,3))
Specificity_matrix=np.zeros((FOLD_N,EPOCH_N,3))

#全データをあらかじめ読み込み
Whole_data=[]
for i in xrange(id):
    fn = file_path[i, 0]
    signal_data = h5py.File(fn, 'r')
    meg_dat = signal_data['meg_dat'].value
    Whole_data.append(meg_dat[:,:])

#for train_index, test_index in skf:
#    print test_index

##conf matrix list
Conf_matrix_list=[]

for train_index, test_index in skf:
    # Model definition
    model = FunctionSet(
        conv1 = F.Convolution2D( 1, 32, (160,64), stride=(1, 2)),
        conv2 = F.Convolution2D(32, 64, (1, 16), stride=(1, 2)),
        conv3=F.Convolution2D(1, 32, (8, 8)),
        conv4=F.Convolution2D(32, 32, (8, 8)),
        conv5=F.Convolution2D(32, 64, (1, 4)),
        conv6=F.Convolution2D(64, 64, (1, 4)),
        conv7=F.Convolution2D(64, 128, (1, 2)),
        conv8=F.Convolution2D(128, 128, (1, 2)),
        conv9=F.Convolution2D(128, 256, (1, 2)),
        conv10=F.Convolution2D(256, 256, (1, 2)),

        fc11 = F.Linear(256*10*1, 1024),
        norm1=F.BatchNormalization(1024),
        fc12 = F.Linear(1024, 1024),
        norm2=F.BatchNormalization(1024),
        fc13 = F.Linear(1024, 3)
    )

    #optimizer = optimizers.MomentumSGD(lr=LR, momentum=0.9)
    #optimizer = optimizers.SMORMS3(lr=LR, eps=1e-16)
    #optimizer = optimizers.AdaGrad(lr=LR)
    optimizer = optimizers.NesterovAG(lr=LR, momentum=0.9)
    optimizer.setup(model)

    if GPU_ID >= 0:
        cuda.get_device(GPU_ID).use()
        model.to_gpu(GPU_ID)

    print 'show train_index,test_index'
    print train_index
    print test_index
    print 'Fold %d ' % (c_f)

    N_test = test_index.shape[0]

    max_accuracy = 0

    #訓練のために所見数を仮想的にupsample
    #まずは各状態の数をカウント
    state_num=[]
    parts=[]
    #NV,EP,PD,SCの順
    for i in xrange(3):
        state_num.append(len(np.where(disease_flag[train_index, 0] == i)[0]))
        parts.append(train_index[disease_flag[train_index, 0] == i])
    max_shoken_num = max(state_num)
    #train indexのupsample
    train_index_upsample=[]
    for i in xrange(3):
        #4は4つの状態を分けているから4にしているだけ
        shou = max_shoken_num // state_num[i]
        amari = max_shoken_num % state_num[i]
        for j in xrange(shou):
            train_index_upsample.append(parts[i])
        train_index_upsample.append(parts[i][0:amari])
    train_index_upsample_array = np.concatenate(train_index_upsample)
    #それぞれnum for eachのデータを取っていくので、それらにidを付ける
    Train_ID = np.arange(train_index_upsample_array.shape[0] * NUM_FOR_EACH)
    Rand_ID = np.random.permutation(Train_ID)

    for epoch in range(EPOCH_N):

        sum_accuracy = 0
        sum_loss = 0

        # Training
        y_pred_train = []
        y_true_train = []

        for i in xrange(0, train_index_upsample_array.shape[0] * NUM_FOR_EACH,BATCH_SIZE):
            X_batch, y_batch = Load_Batch(i,NUM_FOR_EACH,Rand_ID,train_index_upsample_array,BATCH_SIZE,file_path,sample_num,id,Whole_data)

            y_batch=y_batch.astype(np.int32)

            if GPU_ID >= 0:
                X_batch = cuda.to_gpu(X_batch)
                y_batch = cuda.to_gpu(y_batch)

            optimizer.zero_grads()

            loss, accuracy, y_t = forward(X_batch, y_batch,model)
            y_pred_train.append(np.argmax(cuda.to_cpu(y_t.data), axis=1))
            y_true_train.append(cuda.to_cpu(y_batch))
            optimizer.weight_decay(WEIGHT_DECAY)
            loss.backward()
            optimizer.update()

            sum_loss += float(cuda.to_cpu(loss.data)) * len(X_batch)
            sum_accuracy += float(cuda.to_cpu(accuracy.data)) * len(X_batch)

        epoch_acc = sum_accuracy /train_index_upsample_array.shape[0] / NUM_FOR_EACH
        print 'epoch %d | Training lr: 10^%d, loss: %.2f, accuracy: %.4f |' % (epoch, np.log10(optimizer.lr), sum_loss/train_index_upsample_array.shape[0]/NUM_FOR_EACH ,epoch_acc ),
        if epoch_acc>0.85:
            optimizer = optimizers.AdaGrad(lr=LR*0.1)
            optimizer.setup(model)
        # Test
        y_pred = []
        y_true = []
        #所見ごとの判定
        #shoken_prob = []
        shoken_true=[]
        shoken_pred=[]

        #所見ごとに処理する。800sampleずつすべて使うが、余りは切り捨て。
        #後で個々の処理並列化したい
        for i in test_index:
            test_batch_size = int(tmp1[i,0]/800)

            probability_T=[]
            #test_batch_sizeが大きすぎるので分割して処理する
            r_n = test_batch_size // 10
            ama = test_batch_size % 10
            for a in xrange(r_n):
                X_batch_T, y_batch_T = Load_Test_Batch(i, a,10, file_path, disease_flag, Whole_data)
                if GPU_ID >= 0:
                    X_batch_T = cuda.to_gpu(X_batch_T)
                    y_batch_T = cuda.to_gpu(y_batch_T)
                y_T = forward(X_batch_T, y_batch_T, model, train=False)
                probability_T.append(cuda.to_cpu(y_T.data))
            if ama!=0:
                X_batch_T, y_batch_T = Load_Test_Batch(i, r_n, ama, file_path, disease_flag, Whole_data)
                if GPU_ID >= 0:
                    X_batch_T = cuda.to_gpu(X_batch_T)
                    y_batch_T = cuda.to_gpu(y_batch_T)
                y_T = forward(X_batch_T, y_batch_T, model,train=False)
                probability_T.append(cuda.to_cpu(y_T.data))
            probability_T = np.concatenate(probability_T)

            #shoken_prob.append(probability_T)
            #所見ごとの判定
            #probabilityの平均値で判定する（篠崎先生おすすめの方法）
            shoken_true.append(int(disease_flag[i,0]))
            shoken_pred.append(np.argmax(np.mean(probability_T,axis=0), axis=0))

        shoken_pred=np.array(shoken_pred)
        shoken_true=np.array(shoken_true)

        shoken_acc=balanced_accuracy_score(shoken_true,shoken_pred)

        shoken_acc_matrix[c_f,epoch]=shoken_acc

        CM = confusion_matrix(shoken_true, shoken_pred)
        Conf_matrix_list.append(CM)
        for d in xrange(3):
            Sensitivity_matrix[c_f,epoch,d] = 1.0 * CM[d, d] / (CM[d, 0] + CM[d, 1] + CM[d, 2] )
            new = np.delete(CM, d, 0)
            TN = np.delete(new, d, 1)
            Specificity_matrix[c_f,epoch,d] = 1.0 * np.sum(TN) / np.sum(new)

        print 'Test shoken accuracy: %.4f' % (shoken_acc)

    c_f += 1


#model output
model.to_cpu() #モデルをcpuに移す
# Save model npz形式で書き出し
save_model_path = os.path.join(save_dir,'model_5disease_MEG.npz')
serializers.save_npz(save_model_path, model)

#所見ごとの判定結果出力
log_txt += '-----------acc for test shoken----------\n'
for a in xrange(FOLD_N):
    log_txt += '['
    log_txt += 'Fold {0} '.format(a)
    for b in xrange(EPOCH_N):
        log_txt += '{0:f},'.format(shoken_acc_matrix[a,b])
    log_txt += ']\n'

log_txt += '-----------sensitivity for HV----------\n'
for a in xrange(FOLD_N):
    log_txt += '['
    log_txt += 'Fold {0} '.format(a)
    for b in xrange(EPOCH_N):
        log_txt += '{0:f},'.format(Sensitivity_matrix[a,b,0])
    log_txt += ']\n'

log_txt += '-----------sensitivity for EP----------\n'
for a in xrange(FOLD_N):
    log_txt += '['
    log_txt += 'Fold {0} '.format(a)
    for b in xrange(EPOCH_N):
        log_txt += '{0:f},'.format(Sensitivity_matrix[a,b,1])
    log_txt += ']\n'

log_txt += '-----------sensitivity for SC----------\n'
for a in xrange(FOLD_N):
    log_txt += '['
    log_txt += 'Fold {0} '.format(a)
    for b in xrange(EPOCH_N):
        log_txt += '{0:f},'.format(Sensitivity_matrix[a,b,2])
    log_txt += ']\n'

#Specificity
log_txt += '-----------Specificity for HV----------\n'
for a in xrange(FOLD_N):
    log_txt += '['
    log_txt += 'Fold {0} '.format(a)
    for b in xrange(EPOCH_N):
        log_txt += '{0:f},'.format(Specificity_matrix[a, b, 0])
    log_txt += ']\n'

log_txt += '-----------Specificity for EP----------\n'
for a in xrange(FOLD_N):
    log_txt += '['
    log_txt += 'Fold {0} '.format(a)
    for b in xrange(EPOCH_N):
        log_txt += '{0:f},'.format(Specificity_matrix[a, b, 1])
    log_txt += ']\n'

log_txt += '-----------Specificity for SC----------\n'
for a in xrange(FOLD_N):
    log_txt += '['
    log_txt += 'Fold {0} '.format(a)
    for b in xrange(EPOCH_N):
        log_txt += '{0:f},'.format(Specificity_matrix[a, b, 2])
    log_txt += ']\n'

# summaryをテキストファイルに保存
save_filepath = os.path.join(save_dir,'summary.txt')
fid = open(save_filepath,'w')
fid.write(log_txt)
fid.close()

#Conf matrix構築のための配列出力
Conf_matrix_list=np.array(Conf_matrix_list)
save_path_label = os.path.join(save_dir,'conf_matrix_3choice.npy')
np.save(save_path_label, Conf_matrix_list)


