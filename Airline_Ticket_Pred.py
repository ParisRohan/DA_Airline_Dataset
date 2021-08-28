import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

#load dataset
train_data=pd.read_excel(r"C:\Users\Admin\Documents\Coding\Jupyter_Projects\Predict fares of airline tickets/Dataset_Airline.xlsx")
#display loaded data
train_data.head()

#to get sum of missing values 
train_data.isna().sum() 

train_data.dropna(inplace=True)

#get datatype of columns
train_data.dtypes

#get name of columns
train_data.columns

#convert few columns to datetime format as pandas recognized them as object
def change_into_datetime(col):
    train_data[col]=pd.to_datetime(train_data[col])
    
for i in ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']:
    change_into_datetime(i)
    
#extract required data and then drop the column
train_data['Journey Day']=train_data['Date_of_Journey'].dt.day
train_data['Journey Month']=train_data['Date_of_Journey'].dt.month
train_data.drop('Date_of_Journey',axis=1,inplace=True)

def extract_hour(df,col):
    df[col+'_hour']=df[col].dt.hour
    
def extract_min(df,col):
    df[col+'_minute']=df[col].dt.minute

def drop_column(df,col):
    df.drop(col,axis=1,inplace=True)

extract_hour(train_data,'Dep_Time')
extract_min(train_data,'Dep_Time')
drop_column(train_data,'Dep_Time')

extract_hour(train_data,'Arrival_Time')
extract_min(train_data,'Arrival_Time')
drop_column(train_data,'Arrival_Time')

#convert 19h to 19h 00m and 20m to 00h 20m format
duration=list(train_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:
            duration[i]=duration[i]+ ' 00m'
        else:
            duration[i]= '00h '+ duration[i]
            
train_data['Duration']=duration


#split the hours and minutes
def hour(x):
    return x.split(' ')[0][0:-1]
    
def minute(x):
    return x.split(' ')[1][0:-1]

train_data['Duration Hours']= train_data['Duration'].apply(hour)
train_data['Duration Minutes']= train_data['Duration'].apply(minute)

drop_column(train_data,'Duration')

train_data['Duration Hours']=train_data['Duration Hours'].astype(int)
train_data['Duration Minutes']=train_data['Duration Minutes'].astype(int)

train_data.dtypes


#categorical data - nominal data
cat_col=[col for col in train_data.columns if train_data[col].dtype=='O']
cat_col

#categorical data - ordinal data
cont_col=[col for col in train_data.columns if train_data[col].dtype!='O']
cont_col

categorical=train_data[cat_col]
categorical.head()

categorical['Airline'].value_counts()

plt.figure(figsize=(15,5))
sns.boxplot(x=train_data['Airline'],y=train_data['Price'])

#one hot encoding on nominal data
Airline=pd.get_dummies(categorical['Airline'],drop_first=True)
Airline.head() 

Source=pd.get_dummies(categorical['Source'],drop_first=True)
Source.head() 

Destination=pd.get_dummies(categorical['Destination'],drop_first=True)
Destination.head()

#split route
categorical['Route_1']=categorical['Route'].str.split('→').str[0]
categorical['Route_2']=categorical['Route'].str.split('→').str[1]
categorical['Route_3']=categorical['Route'].str.split('→').str[2]
categorical['Route_4']=categorical['Route'].str.split('→').str[3]
categorical['Route_5']=categorical['Route'].str.split('→').str[4]

drop_column(categorical,'Route')
categorical.head()

#to get the count of missing values
categorical.isnull().sum()  

# output of above code shows that 'Route_3','Route_4','Route_5' have null values
for i in ['Route_3','Route_4','Route_5']:
    categorical[i].fillna('None',inplace=True)
    
for i in categorical.columns:
    print('{} has total {} categories'.format(i,len(categorical[i].value_counts())))

#'Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5' have a lot of categories. So we will encode them     
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5']:
    categorical[i]=encoder.fit_transform(categorical[i])
    
#to get count of unique itmes in a column
categorical['Additional_Info'].nunique()

#to get count of unique itmes and their name in a column
pd.value_counts(categorical['Additional_Info'])

#to get name of unique itmes in a column
categorical['Additional_Info'].unique()

drop_column(categorical,'Additional_Info')

#to encode Total_Stops column
categorical['Total_Stops'].unique()

#create a dictionary for mapping
dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}
categorical['Total_Stops']=categorical['Total_Stops'].map(dict)
categorical.head()

#concat all the df
data_train=pd.concat([categorical,Airline,Source,Destination,train_data[cont_col]],axis=1)
data_train.head()

drop_column(data_train,'Airline')
drop_column(data_train,'Source')
drop_column(data_train,'Destination')

pd.set_option('display.max_columns',35)
data_train.head()

#handle outliers
def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)

plot(data_train,'Price')

data_train['Price']=np.where(data_train['Price']>=40000,data_train['Price'].median(),data_train['Price'])

#seperate price from rest of data
X=data_train.drop('Price',axis=1)
X.head()

Y=data_train['Price']
Y.head()

#Feature selection
from sklearn.feature_selection import mutual_info_classif

imp=pd.DataFrame(mutual_info_classif(X,Y),index=X.columns)

imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

from sklearn import metrics

def predict(ml_model):
    model=ml_model.fit(x_train,y_train)
    print('Training Score: {}'.format(model.score(x_train,y_train)))
    y_predict=model.predict(x_test)
    print('Prediction are: \n{}'.format(y_predict))
    print('\n')
    
    r2_score=metrics.r2_score(y_test,y_predict)
    print('r2 score is: {}'.format(r2_score))
    
    print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_predict))
    print('Mean Squared Error: ',metrics.mean_squared_error(y_test,y_predict))
    print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
    
    sns.distplot(y_test-y_predict)
    
from sklearn.ensemble import RandomForestRegressor as RFG
predict(RFG())

from sklearn.linear_model import LinearRegression as LR
predict(LR())

from sklearn.neighbors import KNeighborsRegressor as KNN
predict(KNN())

from sklearn.tree import DecisionTreeRegressor as DTR
predict(DTR())

