# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Code starts here
df = pd.read_json(path,lines=True)
df.columns=df.columns.str.strip().str.lower().str.replace(' ', '_')
##df.dropna(inplace=True)
df.drop(columns=['waist', 'bust', 'user_name','review_text','review_summary','shoe_size','shoe_width'],axis=1,inplace=True)
print(df.columns)
y=df['fit']
X=df.drop('fit',axis=1)
print(X.columns)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=6)
print(y_train.shape)
print(y_test.shape)
# Code ends here


# Code ends here


# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)


# Code starts here
g_by_category = df.groupby('category')
cat_fit=g_by_category.count()['fit']



# Code ends here


# --------------
# Code starts here
g_by_category = df.groupby('category')
cat_len = g_by_category['length'].value_counts()
print(cat_len)
cat_len = cat_len.unstack()
print(cat_len)
plot_barh(cat_len, 'length')

# Code ends here


# --------------
# Code starts here
def get_cms(x):
    if type(x) == type(1.0):
        return

    try: 
        return (int(x[0])*30.48) + (int(x[4:-2])*2.54)
    except:
        return (int(x[0])*30.48)       


print(X_train['height'].isnull().sum())
###X_train['height'].dropna(inplace=True)

##for i in range(len(X_train)):
##    X_train['height'][i]= (int(X_train['height'][i][0])*30.48) + (int(X_train['height'][i][4:-2])*2.54)
X_train['height']=X_train['height'].apply(get_cms)
X_test['height']=X_test['height'].apply(get_cms)


# Code ends here


# --------------
# Code starts here
print(X_train.isnull().sum())
X_train.dropna(subset=['height','length','quality'],inplace=True)
X_test.dropna(subset=['height','length','quality'],inplace=True)
X_train['bra_size'].fillna((X_train['bra_size'].mean()), inplace=True)
X_test['bra_size'].fillna((X_test['bra_size'].mean()), inplace=True)
mode_1=X_train['cup_size'].mode()[0]
mode_2=X_test['cup_size'].mode()[0]
X_train['cup_size']=X_train['cup_size'].replace(np.nan,mode_1)
X_test['cup_size']=X_test['cup_size'].replace(np.nan,mode_2)


# Code ends here


# --------------
print(X_test.head(2))

X_train=pd.get_dummies(data=X_train,columns=["category", "cup_size","length"],prefix=["category", "cup_size","length"])
X_test=pd.get_dummies(data=X_test,columns=["category", "cup_size","length"],prefix=["category", "cup_size","length"])



# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score




# code starts here 

# Instantiate logistic regression
model = DecisionTreeClassifier(random_state = 6)

# fit the model
model.fit(X_train,y_train)

# predict the result
y_pred =model.predict(X_test)

# calculate the f1 score
score = accuracy_score(y_test, y_pred)
print(score)

# calculate the precision score
precision = precision_score(y_test, y_pred, average=None)
print(precision)






# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

# Code starts here
model=DecisionTreeClassifier(random_state=6)
grid=GridSearchCV(estimator=model,param_grid=parameters)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


