import sklearn.datasets as data
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = data.load_iris()
x,X,y,Y = train_test_split(data.data,data.target,test_size=0.2)
print(y)
classf = SVC()
classf.fit(x,y)
print(accuracy_score(Y,classf.predict(X))*100)



