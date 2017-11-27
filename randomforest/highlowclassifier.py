import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Importing Training Data...")
train_data = pd.read_table("../input/train.tsv")
highlowprice = 100
target = np.array([(0 if x>highlowprice else 1) for x in train_data.price])

# Initial Columns:
# train_id, name, item_condition_id, category_name, brand_name,
# price, shipping, item_description

train_data = train_data[['name','item_condition_id','category_name','brand_name']]
train_data = train_data.fillna('missing')

print("Encoding Data...")
encoders = {}
for c in train_data.columns:
    encoders[c] = LabelEncoder()
    encoders[c].fit(train_data[c].values)
    train_data[c] = encoders[c].transform(train_data[c].values)

ts = [int(0.5*len(target)),int(0.8*len(target))]
Xtrain,Xval,Xtest = train_data.values[:ts[0]],\
        train_data.values[ts[0]:ts[1]],\
        train_data.values[ts[1]] 
ytrain,yval,ytest = target[:ts[0]],target[ts[0]:ts[1]],target[ts[1]]

print("Training Model...")
clf = RandomForestClassifier()
clf.fit(Xtrain,ytrain)
featimportance = clf.feature_importances_
ypred = clf.predict(Xval)

accuracy = np.sum([i==j for i,j in zip(ypred,yval)])/len(yval)
low = [(i,j) for i,j in zip(ypred,yval) if j==1]
lowacc = np.sum([i==j for i,j in low])/len(low)
high = [(i,j) for i,j in zip(ypred,yval) if j==0]
highacc = np.sum([i==j for i,j in high])/len(high)
print('Overall Accuracy: ',accuracy)
print('Low Price Accuracy: ',lowacc)
print('High Price Accuracy: ',highacc)

print("Feature Importance:")
for f,v in zip(train_data.columns,featimportance):
    print(' ',f,': ',v)
