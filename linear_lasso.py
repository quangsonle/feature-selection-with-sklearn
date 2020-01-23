import numpy as np
import pandas as pd

# Read CSV
task_file = 'task_data.csv'
with open(task_file, mode = 'rb') as f:
    data = pd.read_csv(f)
    
# Extract names of each column (using pandas)
headers = np.array(list(data.columns.values))
names = headers[2:]
#print ("Feature names shape is {}".format(names.shape))

# Extract features (using pandas and numpy)
np_array = data.as_matrix()
X = np_array[:,2:]
#print ("Features shape is {}".format(X.shape))

# Extract labels (using pandas)
Y = data['class_label'].as_matrix()
#print ("Labels shape is {}".format(Y.shape))
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
rf = Lasso(alpha=10)
rf.fit(X,Y)
rank = sorted(zip(map(lambda x: round(x, 2), 
                      rf.coef_), names), reverse=True)
print("Features and Predictive Scores:")
for el in rank: print(el)
    
  