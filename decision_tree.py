import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt
iris = datasets.load_iris()
#   f = open('decision_tree_data.txt')
x_train = []
y_train = []
for line in iris['data']:
    line = np.asarray(line,dtype = np.float32)
    x_train.append(line[:-1])
    y_train.append(line[:-1])
x_train = np.asmatrix(x_train)
y_train = np.asmatrix(y_train)
model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image
dot_data = StringIO()
tree.export_graphviz(model, out_file=dot_data,  
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
(graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())




%windir%\System32\cmd.exe "/K" C:\Users\dostrows\AppData\Local\Continuum\Anaconda3-5.2.0\Scripts\activate.bat C:\Users\dostrows\AppData\Local\Continuum\Anaconda3-5.2.0