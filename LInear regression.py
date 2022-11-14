import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train=torch.FloatTensor([[1],[2],[3]])
y_train=torch.FloatTensor([[4],[5],[6]])

w=torch.zeros(1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)
optimizer=optim.SGD([w,b],lr=0.01)

#손실 함수가 최소가 되는 가중치와 편향을 찾는다 (=경사하강법)
for epoch in range(3001):
  #H(x)
  hypothesis=x_train*w+b
  
  #cost 함수
  cost=torch.mean((hypothesis-y_train)**2)

  optimizer.zero_grad() #활성화 함수 초기화
  cost.backward() #cost를 미분하여 기울기 계산
  optimizer.step()

  if epoch%100==0:
    print('Epoch : {:4d}/{} w : {:.3f}, b : {:.3f}, Cost : {:.6f}'.format(epoch, 3000, w.item(),b.item(),cost.item()))
