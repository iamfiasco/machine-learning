from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.datasets as data
import sys
data = data.load_iris()
x,X,y,Y = train_test_split(data.data,data.target,test_size=0.2)
classf = KNeighborsClassifier(n_neighbors=int(sys.argv[1]))
classf.fit(x,y)
print(accuracy_score(Y,classf.predict(X))*100)
