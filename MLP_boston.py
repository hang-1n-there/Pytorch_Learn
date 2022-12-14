from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #전체 데이터를 학습데이터와 평가데이터로 나눔
from sklearn.preprocessing import MinMaxScaler #데이터 내의 값을 0~1로 조정함

import torch
from torch import nn,optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt #시각화

bos = load_boston()

#bos는 data feature name 등 다양한 정보를 포함하고 있다.
df=pd.DataFrame(bos.data) #데이터의 변수를 데이터프레임으로 만들기
df.columns=bos.feature_names #데이터의 변수명 불러오기
df['Price']=bos.target #집 값에 해당하는 타겟 값 불러오기

df.head(10) #인수가 없으면 상위 5줄만 보여줌

#최댓값을 1로, 최솟값을 0으로 바꿔서 사잇값을 0~1사이로 바꾼다.
X=df.drop('Price', axis=1).to_numpy() #데이터프레임에서 타겟값 Price만 제외하고 모두 numpy 배열로 만듬
Y=df['Price'].to_numpy().reshape((-1,1))

scaler=MinMaxScaler()
scaler.fit(X) #열을 기준으로 minmaxscaler를 진행한다
X=scaler.transform(X)

scaler.fit(Y)
Y=scaler.transform(Y)

#텐서 데이터로 변환
class TensorData(Dataset):
  def __init__(self,x_data,y_data):
    self.x_data=torch.FloatTensor(x_data)
    self.x_data=torch.FloatTensor(y_data)
  def __getitem__(self,index):
    
    return self.x_data[index], self.y_data[index]
  def __len__(self):
    
    return self.len