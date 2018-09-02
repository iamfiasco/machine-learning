import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
#preprocessing the datasets
name='data_batch_'
#list all batches
files = glob.glob(name+'*')
data=[]
#merge all batches
for i in files:
	data.append(pickle.load(open(i,'rb'),encoding='bytes'))
datas=data[0][b'data']
for i in range(1,5):
	datas=np.concatenate([datas,data[i][b'data']])
print(datas.shape)
#generate random index
import random
index=random.randint(0,50000)
#transposing and reshaping the value to a form which matplotlib can understand
plt.imshow(np.transpose(datas[index].reshape(3,32,32),(1,2,0)))
plt.show()
