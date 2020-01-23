import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit

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

rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))
rank =sorted(scores, reverse=True)
for el in rank: print(el)



