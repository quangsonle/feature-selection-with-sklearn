import numpy as np
import pandas as pd
from stability_selection import StabilitySelection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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
lambda_grid = np.linspace(0.001, 0.5, num=100)
base_estimator = Pipeline([('scaler', StandardScaler()),('model', LogisticRegression(penalty='l1'))])
selector = StabilitySelection(base_estimator=base_estimator, lambda_name='model__C',lambda_grid=np.logspace(-5, -1, 50))
selector.fit(X, Y)                                      
selected_variables = selector.get_support(indices=True,threshold=0.00001)
selected_scores = selector.stability_scores_.max(axis=1)

print('Ranking is:')
print('-----------------------')

#for idx, (variable, score) in enumerate(zip(selected_variables, selected_scores[selected_variables])):
# print('Variable %d: [%d], score %.3f' % (idx + 1, variable, score))
rank = sorted(zip(selected_scores,names),reverse=True)
for el in rank: print(el)




