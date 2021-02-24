#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import files
uploaded = files.upload()   #loading the train dataset


# ## Importing the KDDTrain+ and KDDTest+ dataset files in colab

# In[ ]:


from google.colab import files
uploaded = files.upload()   #loading the test dataset


# In[1]:


import pandas as pd  #importing the necessary packages
import warnings
warnings.filterwarnings("ignore")
import shutil
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
#from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
#from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import f1_score, make_scorer
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use(u'nbAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
import pandas as pd
from multiprocessing import Process# this is used for multithreading
import multiprocessing
import codecs# this is used for file operations 
import random as r
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Loading the data into dataframes

# In[2]:


df_train=pd.read_csv("C:/NSL-KDD/KDDTrain+.txt")
df_test=pd.read_csv("C:/NSL-KDD/KDDTest+.txt")


# In[ ]:


df_train.head() #checking the first 5 rows of the data


# In[ ]:


df_train.tail() #checking the last 5 rows of the data


# In[ ]:


print("Number of data points in train data", df_train.shape)  #checking the shape and the total number of attributes of the data
print('-'*50)
print("The attributes of data :", df_train.columns.values)


# In[ ]:


print("Number of data points in test data", df_test.shape)
print(df_test.columns.values)
df_test.head(2)


# In[ ]:


df_train.dtypes  #checking the datatypes of all the columns


# In[3]:


#renaming the columns

df_train = df_train.rename(columns={"0":"Duration","tcp":"protocol_type","ftp_data":"service","SF":"flag","491":"src_bytes",
                                    "0.1":"dest_bytes","0.2":"Land","0.3":"wrong_fragment","0.4":"Urgent packets","0.5":"hot",
                                    "0.6":"num_failed_logins","0.7":"logged_in","0.8":"num_compromised","0.9":"root_shell",
                                    "0.10":"su_attempted","0.11":"num_root","0.12":"num_file_creations","0.13":"num_shells",
                                    "0.14":"num_access_files","0.15":"num_outbound_cmds","0.16":"is_host_login","0.17":"is_guest_login",
                                    "2":"count","2.1":"srv_count","0.00":"serror_rate","0.00.1":"srv_serror_rate","0.00.2":"rerror_rate",
                                    "0.00.3":"srv_rerror_rate","1.00":"same_srv_rate","0.00.4":"diff_srv_rate","0.00.5":"srv_diff_host_rate",
                                    "150":"dst_host_count","25":"dst_host_srv_count","0.17.1":"dst_host_same_srv_rate",
                                    "0.03":"dst_host_diff_srv_rate","0.17.2":"dst_host_same_src_port_rate",
                                    "0.00.6":"dst_host_srv_diff_host_rate","0.00.7":"dst_host_serror_rate",
                                    "0.00.8":"dst_host_srv_serror_rate","0.05":"dst_host_rerror_rate","0.00.9":"dst_host_srv_rerror_rate",
                                    "normal":"attack_type","20":"Score"})


# In[ ]:
print(df_train['protocol_type'].value_counts())
print(df_train['flag'].value_counts())

df_train.head()  #checking the dataframe after renaming


# In[4]:


#renaming the columns of test data

df_test = df_test.rename(columns={"0":"Duration","tcp":"protocol_type","private":"service","REJ":"flag","0.1":"src_bytes",
                                    "0.2":"dest_bytes","0.3":"Land","0.4":"wrong_fragment","0.5":"Urgent packets","0.6":"hot",
                                    "0.7":"num_failed_logins","0.8":"logged_in","0.9":"num_compromised","0.10":"root_shell",
                                    "0.11":"su_attempted","0.12":"num_root","0.13":"num_file_creations","0.14":"num_shells",
                                    "0.15":"num_access_files","0.16":"num_outbound_cmds","0.17":"is_host_login","0.18":"is_guest_login",
                                    "229":"count","10":"srv_count","0.00":"serror_rate","0.00.1":"srv_serror_rate","1.00":"rerror_rate",
                                    "1.00.1":"srv_rerror_rate","0.04":"same_srv_rate","0.06":"diff_srv_rate","0.00.2":"srv_diff_host_rate",
                                    "255":"dst_host_count","10.1":"dst_host_srv_count","0.04.1":"dst_host_same_srv_rate",
                                    "0.06.1":"dst_host_diff_srv_rate","0.00.3":"dst_host_same_src_port_rate",
                                    "0.00.4":"dst_host_srv_diff_host_rate","0.00.5":"dst_host_serror_rate",
                                    "0.00.6":"dst_host_srv_serror_rate","1.00.2":"dst_host_rerror_rate","1.00.3":"dst_host_srv_rerror_rate",
                                    "neptune":"attack_type","21":"Score"})


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


# Rows containing duplicate data
duplicate_rows_df = df_train[df_train.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[ ]:


# Finding the null values.
print(df_train.isnull().sum())


# In[5]:


label_encoder1 = preprocessing.LabelEncoder() 
df_train['protocol_type']= label_encoder1.fit_transform(df_train['protocol_type']) 
a=label_encoder1.classes_


# In[8]:


int_features=['tcp','private','REJ']


# In[10]:


int_features


# In[9]:


for i in range(len(a)):
        if a[i]==int_features[0]:
            int_features[0]=i


# In[11]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)


# In[12]:


@app.route('/')
def home():
    if request.method == "POST": 
    
        return render_template("index.html") 

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]


# # Exploratory Data Analysis

# In[ ]:


y_value_counts = df_train['attack_type'].value_counts()  #checking the distribution of different class of the class label


# Ploting the bar plot of attack type variable to check the distribution of different class in the dataset-Train

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
y_value_counts.plot(kind="bar", fontsize=10)


# Ploting the bar plot of attack type variable to check the distribution of different class in the dataset-Test

# In[ ]:


y_test_value_counts = df_train['attack_type'].value_counts()  
get_ipython().run_line_magic('matplotlib', 'inline')
y_test_value_counts.plot(kind="bar", fontsize=10)


# Observation: The above plot clearly shows that the attack type "normal" has the highest distribution in the data followed by "neptune" and then the other classes whose value count is very less compared to these two classes. The distribution is almost same for both test dataset and train dataset.

# In[ ]:


counter = Counter(df_train['attack_type'])
a=dict(counter)
per=[]
for k,v in counter.items():
	per.append(v / len(df_train['attack_type']) * 100) #calculating the percentage distribution of my class label


# ## Plotting the pie chart of attack type with the percentage distribution of each attack type 

# In[ ]:


patches, texts = plt.pie(per, startangle=90, radius=2)  #https://stackoverflow.com/questions/23577505/how-to-avoid-overlapping-of-labels-autopct-in-a-matplotlib-pie-chart
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(a.keys(), per)]
patches, labels, dummy =  zip(*sorted(zip(patches, labels, per),
                                          key=lambda x: x[2],
                                          reverse=True))

plt.legend(patches,labels , loc='left center', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)

plt.savefig('piechart.png', bbox_inches='tight')


# The above plot gives an idea of the percentage value of each class. The normal class covers almost 53% of the data followed by neptune class which covers 32% and then the rest of the classes each covering less than 3% of the entire dataset. From the above plot we can conclude that our dataset is an imbalanced dataset with huge difference in the distribution of different class labels

# Lets have a look at the distribution of each feature of the dataframe.
# 
# ---
# 
# ---
# 
# 
# 
# 

# In[ ]:


df_train.hist(figsize=(35,35)) 
plt.tight_layout()
plt.show()


# ## Now lets view the correlation between features and target variable.

# In[ ]:




# In[ ]:


import phik
from phik import resources, report
corr_matrix=df_train.phik_matrix()
corr_matrix


# In[ ]:


print(corr_matrix["attack_type"].sort_values(ascending=False)[1:])


# In[ ]:


corr = corr_matrix["attack_type"].sort_values(ascending=False)


# Hence we can conclude that the features which has strong correlation with the target variables are 
# 
# 1.  wrong_fragment
# 2.  Land
# 3.  service
# 4.  protocol_type
# 5.  logged_in
# 6.  flag
# 7.  Score
# 8.  count
# 9.  dst_host_srv_diff_host_rate
# 10. same_srv_rate
# 11. dst_host_serror_rate
# 12. serror_rate
# 13. dst_host_srv_serror_rate
# 14. serror_rate
# 
# 
# 

# The above figure covers the rest all continous variables of our dataset and their individual distribution. The histograms shows for most of the features the data is unevenly distributed and one value dominates in all of them. Also we donot see any concrete observation from the distributions like whether the features are gaussian distributed or not. Hence no effective analysis can be made from the univariate analysis except the fact that the data's are unevenly distributed. Lets move to the Bivariate analysis and check relationship between features

# # **Univariate analysis**

# Here we will check the distribution of features which has strong correlation with target variable along with the class label.

# ## 1.Plot between wrong fragment and attack types

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(25.7,5.27)})
sns.barplot(x='attack_type',y='wrong_fragment',data=df_train)


# In[ ]:


df_train['wrong_fragment'].value_counts()


# Observation: Most of the records with wrong_fragment value "3" and "1" belongs to attack type "teardrop" and "pod". Now lets remove these types from the target class and visualize the distribution again

# In[ ]:


df_dash= df_train[(df_train['attack_type'] != 'teardrop') & (df_train['attack_type'] != 'pod')]


# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(25.7,5.27)})
sns.barplot(x='attack_type',y='wrong_fragment',data=df_dash)


# Observation: We can say that rest all the datapoints belongs to wrong_fragment "0" and belongs to rest of the attack types.

# Now lets visualize the same for test dataset.

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(35.7,5.27)})
sns.barplot(x='attack_type',y='wrong_fragment',data=df_test)


# Here also we can observe the same distribution. Most datpoints with wrong fragment not equal to 0 belongs to attack type "teardrop" and "pod".

# ## 2. Plot between land and attack type

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(25.7,5.27)})
sns.barplot(x='attack_type',y='Land',data=df_train)


# In[ ]:


df_train['Land'].value_counts()


# Observation: As our feature is binary we can say that the datapoints with Land value 1 belongs to only attack type "land". We will visualize the distribution again after removing the "land" type from class label.

# In[ ]:


df_dash= df_train[df_train['attack_type'] != 'land']


# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(25.7,5.27)})
sns.barplot(x='attack_type',y='Land',data=df_dash)


# Observation: Attack type "normal" covers majority of the datapoints with Land value "0". Lets check the same for test dataset.

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(35.7,5.27)})
sns.barplot(x='attack_type',y='Land',data=df_test)


# Observation: We can witness the same distribution of Land feature with attack type "land" in the test dataset as well.

# ## 3. Plot between Service and attack type

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(25.7,25.27)})
sns.stripplot(x=df_train['attack_type'],y=df_train['service'])


# Observation : From the above plot we can say the different service types are distributed among all the attack types where "normal","neptune" and "portsweep" covers majority of the service types.

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(25.7,25.27)})
sns.stripplot(x=df_test['attack_type'],y=df_test['service'])


# Observation: The same can be seen for test dataset as well however we can conclude that attack type "normal" covers all the service types.

# ## 4. Plot between protocol_type and attack type

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(20.7,8.27)})
sns.countplot(x='protocol_type',hue="attack_type",data=df_train)


# Observation: From the above plot we can observe that the protocol type "tcp" covers majority of the datapoints and is distributed between attack type "normal" and "neptune". We will visualize the same for test dataset.

# In[ ]:



sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(30.7,8.27)})
sns.countplot(x='protocol_type',hue="attack_type",data=df_test,palette = "Set2")


# Observation: In the test dataset we can observe the same distribution. The protocol type "tcp" covers majority of the dataset with maximum distribution of attack type "normal" followed by "neptune". 

# ## 5. Plot between logged_in and attack type

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(25.7,5.27)})
sns.barplot(x='attack_type',y='logged_in',data=df_train)


# Observation: From the above plot we can see that the logged in feature is distributed among multiple attack types. Our feature is binarry where both the classes are almost equally distributed, we can see that attack type "warezclient","back","overflow" and "phf" are all covering the logged in value 1. We will visualize the same for test dataset.

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(30.7,5.27)})
sns.barplot(x='attack_type',y='logged_in',data=df_test)


# Observation: The same distribution of attack types can be observed from the test dataset as well.

# ## 6. Plot between flag and attack type

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(30.7,16.27)})
sns.countplot(x='flag',hue="attack_type",data=df_train,palette = "Set2")


# Observation: From the above plot we can say that the flag type "SF" covers the majority of the distribution followed by "SO" and "REJ". Attack type "normal" covers majority of the datapoints with falg type "SF". We will visualize the same for test dataset.

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(30.7,16.27)})
sns.countplot(x='flag',hue="attack_type",data=df_test,palette = "Set2")


# Observation : The test dataset also follows the same distribution as the train data with majority of datapoints belonging to flag type "SF".

# ## 7. Plot between score and attack type

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(30.7,5.27)})
sns.barplot(x='attack_type',y='Score',data=df_train)


# Observation: From the above plot we can conclude that most of the attack types belongs to score value more than 15. We will check the same for test dataset as well.

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(35.7,5.27)})
sns.barplot(x='attack_type',y='Score',data=df_test)


# Observation: The dataset follows the same distribution as train dataset.

# ## 8. Plot between count and attack type

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(35.7,5.27)})
sns.barplot(x='attack_type',y='count',data=df_train)


# Observation: From the above plot we can see that most of the attack types belongs to count value of less than 5. Attack type "smurf" belongs to count value greater than 350 followed by "neptune". 

# In[ ]:


sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(40.7,5.27)})
sns.barplot(x='attack_type',y='count',data=df_test)


# Observation: The test data follows a different distribution for feature "count". The attack type "satan buffer" covers the highest count values(>300) followed by "smurf","saint","neptune".

# # Bivariate Analysis

# # Next we will check the correlation between different columns using heatmap

# In[ ]:


sns.set(rc={'figure.figsize':(25,25)})
sns.heatmap(corr_matrix);


# In[ ]:


corr_matrix


# Observation: We can figure out the features which are strongly correlated with each other from the above matrix:
# 1. Protocol type and Service are highly correlated.
# 2. is_guest_login and Service are highly correlated.
# 3. dst_host_same_srv_rate and Service are highly correlated.
# 4. Count and logged_in are correlated.
# 5. num_compromised and num_root are highly correlated.
# 6. dst_host_same_srv_rate and dst_host_srv_count are highly correlated.

# # Plotting box plot for few of the features to check for outliers.

# 1. Destination Bytes

# In[ ]:


sns.set(rc={'figure.figsize':(10,8)})
sns.boxplot(x=df_train['dest_bytes'])


# The above box plot is highly skewed and also there are outliers in our data.

# 2. Hot

# In[ ]:


sns.set(rc={'figure.figsize':(10,8)})
sns.boxplot(x=df_train['hot'])


# We can observe a lot of outliers for this feature.

# 3.count

# In[ ]:


sns.set(rc={'figure.figsize':(10,8)})
sns.boxplot(x=df_train['count'])


# The data ranges between 0 and 350. Anything apart from that can be considered as outlier

# # Applying TSNE algorithm to check the distribution of data in lower dimensional space.

# In[ ]:


X = df_train.drop(['attack_type','Score','protocol_type','service','flag'], axis=1)
y=df_train['attack_type']


# In[ ]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


tsne = TSNE(random_state=17)
tsne_repr = tsne.fit_transform(X_scaled)


# In[ ]:


df_train['tsne1']=tsne_repr[:, 0]


# In[ ]:


df_train['tsne2']=tsne_repr[:, 1]


# In[ ]:


color_dict = dict({'normal': 'blue', 'neptune': 'orange','satan': 'yellow','ipsweep':'snow','portsweep':'black','smurf':'red','nmap':'aqua','back':'indigo',
                   'teardrop':'wheat','warezclient':'purple','pod':'slateblue','guess_passwd':'pink','buffer_overflow':'ivory',
                   'warezmaster':'maroon','land':'brown','imap':'cyan','rootkit':'olive',
                   'loadmodule':'lavender','ftp_write':'dimgrey','multihop':'royalblue','phf':'lime','perl':'seagreen','spy':'bisque'})


# In[ ]:


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne1", y="tsne2",
    hue="attack_type",
    palette=color_dict,
    data=df_train,
    legend="full",
    
)


# Observation:we can conclude that the few attack types like normal, neptune,imap are fairly seperable. We do not see much overalap among most of the categories.

# In[ ]:


df_train=df_train.drop(['tsne1','tsne2'],axis=1)


# ## Applying PCA on the dataset

# In[ ]:


pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)


# In[ ]:


df_train['pca1']=pca_result[:, 0]
df_train['pca2']=pca_result[:, 1]
df_train['pca3']=pca_result[:, 2]


# In[ ]:


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca1", y="pca2",
    hue="attack_type",
    palette=color_dict,
    data=df_train,
    legend="full",
    alpha=0.3
)


# Observation:From PCA as well we can say that the class label normal and neptune are seperable in lower dimension space however other classes are overlapping..

# In[ ]:


df_train=df_train.drop(['pca1','pca2','pca3'],axis=1)


# ## Conclusion: The dataset has few features which can be dropped and we can get rid of the outliers as well however I would like to proceed with a base model taking all the features and datapoints and evaluate the performance of it. Based on that I can then proceed with feature selection and dropping the outliers as it can be evident if the impact of outliers in the data is considerate or not.
# 

# As the different attack types belong to 4 major attack categories, lets assign the attack types to these 4 types.
# 
# 

# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.6760&rep=rep1&type=pdf#:~:text=The%20data%20in%20NSL%2DKDD,in%20weka%20tool%20%5B11%5D.

# ![attack.PNG](data:image/png;base64,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)

# In[6]:


attack_sep={'normal':"Normal",'neptune':"DOS",
            'satan':"Probe",'ipsweep':"Probe",'named':"R2L",
            'ps':"U2R",'sendmail':"R2L",'xterm':"U2R",'xlock':"R2L",
            'xsnoop':"R2L",'udpstorm':"DOS",'sqlattack':"U2R",'worm':"DOS",'portsweep':"Probe",
            'smurf':"DOS",'nmap':"Probe",'back':"DOS",'mscan':"Probe",'apache2':"DOS",'processtable':"DOS",
            'snmpguess':"R2L",'saint':"Probe",'mailbomb':"DOS",'snmpgetattack':"R2L",'httptunnel':"R2L",'teardrop':"DOS",
            'warezclient':"R2L",'pod':"DOS",'guess_passwd':"R2L",'buffer_overflow':"U2R",'warezmaster':"R2L",'land':"DOS",'imap':"R2L",
            'rootkit':"U2R",'loadmodule':"U2R",'ftp_write':"R2L",'multihop':"R2L",'phf':"R2L",'perl':"U2R",'spy':"R2L"}


# In[7]:


df_train.replace({'attack_type':attack_sep},inplace=True)


# In[8]:


df_test.replace({'attack_type':attack_sep},inplace=True)


# # Lets train a base model on the entire dataset and evaluate the performance.

# In[9]:


from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder() 

df_train['protocol_type']= label_encoder.fit_transform(df_train['protocol_type']) 
df_test['protocol_type']= label_encoder.transform(df_test['protocol_type']) 

label_encoder = preprocessing.LabelEncoder() 
  
df_train['service']= label_encoder.fit_transform(df_train['service']) 
df_test['service']= label_encoder.transform(df_test['service']) 

label_encoder = preprocessing.LabelEncoder() 

df_train['flag']= label_encoder.fit_transform(df_train['flag']) 
df_test['flag']= label_encoder.transform(df_test['flag']) 


# In[10]:


y=df_train['attack_type']
X=df_train.drop(['attack_type'],axis=1)

y_test=df_test['attack_type']
X_test=df_test.drop(['attack_type'],axis=1)


# In[11]:


sc = StandardScaler()  #standardizing the data
X_train = sc.fit_transform(X)
X_test = sc.transform(X_test)


# # Lets build a base model on our dataset

# In[18]:


def falseposrate(conf_matrix,y_test,pred):
  FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix) 
  FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
  TP = np.diag(conf_matrix)
  TN = conf_matrix.sum() - (FP + FN + TP)
  FP = FP.astype(float)
  FN = FN.astype(float)
  TP = TP.astype(float)
  TN = TN.astype(float)
  FPR = FP/(FP+TN)
  recall = recall_score(y_test, pred,average='micro')
  precision = precision_score(y_test, pred,average='micro')
  return FPR,recall,precision


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
clf= svm.SVC(kernel='linear',probability=True)
clf.fit(X_train,y)
pred = clf.predict(X_test)
recall = recall_score(y_test, pred,average='micro')
precision = precision_score(y_test, pred,average='micro')
score = metrics.accuracy_score(y_test, pred)
f1score= f1_score(y_test, pred, average='micro')
print("Accuracy :",score)
print('=' * 50)
print("F1 score :",f1score)

cnf_matrix = confusion_matrix(y_test, pred)
#sns.heatmap(cnf_matrix)
fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(clf, X_test, y_test,ax=ax,cmap=plt.cm.Blues)
plt.show()

print('_' * 50)
print(cnf_matrix)

FPR= falseposrate(cnf_matrix)
print('=' * 50)
print("|False positive Rate :|")
print('=' * 50)
print(FPR)
print('=' * 50)
print("|Precision:|")
print('=' * 50)
print(precision)
print('=' * 50)
print("|recall:|")
print('=' * 50)
print( recall)
print('=' * 50)
print("|Classification report|")
print('=' * 50)
print(metrics.classification_report(y_test,pred))


# Our model is giving 78% F1 score. Lets see if we can improve the performnace.

# # Lets try to improve our model performance.

# # Lets apply some feature selection techniques on our datasets and analyze the performance.

# ## Coorelation based feature Selection( CFS)

# In[ ]:


abs_corr = abs(corr)
relevant_features = abs_corr[abs_corr>0.5]


# Lets look at the features which are highly correlated with each other.

# In[ ]:


new_df= df_train[relevant_features.index]
new_df_test= df_test[relevant_features.index]


# In[ ]:


y_cfs=new_df['attack_type']
X_cfs=new_df.drop(['attack_type'],axis=1)

y_test_cfs=new_df_test['attack_type']
X_test_cfs=new_df_test.drop(['attack_type'],axis=1)


# In[ ]:


sc = StandardScaler()
X_train_cfs = sc.fit_transform(X_cfs)
X_test_cfs = sc.transform(X_test_cfs)


# In[31]:




# In[35]:


from catboost import CatBoostClassifier


# In[ ]:


def training(clf,xtrain,xtest,ytrain,ytest,attack_type):
    print('\n')
    print('=' * 50)
    print("Training ",attack_type)
    print(clf)
    clf.fit(xtrain, ytrain)
    print('_' * 50)
    pred = clf.predict(xtest)
    print('_' * 50)
    roc = roc_auc_score(ytest, clf.predict_proba(xtest), multi_class='ovo', average='weighted')
    score = metrics.accuracy_score(ytest, pred)
    f1score= f1_score(ytest, pred, average='micro')
    print("accuracy:   %0.3f" % score)
    print()
    print('_' * 50)
    print("|classification report|")
    print('_' * 50)
    print(metrics.classification_report(ytest, pred))
    print('_' * 50)
    print("confusion matrix:")
    print(metrics.confusion_matrix(ytest, pred))
    cm= metrics.confusion_matrix(ytest, pred)
    print()
    print('_' * 50)
    print("ROC AUC Score :",roc)
    FPR,precision,recall= falseposrate(cm,ytest,pred)
    print('_' * 50)
    print("False Positive Rate is :",FPR)
    clf_descr = str(clf).split('(')[0]
    return clf_descr,score, f1score,roc,FPR,precision,recall


# In[ ]:


results_CFS= []

for clf, name in (
        (GaussianNB() ,"Naive Bayes"),
        (KNeighborsClassifier(n_neighbors = 7),"KNN"),
        (OneVsRestClassifier(svm.SVC(probability=True)),"One vs Rest SVM "),
        (RandomForestClassifier(), "Random forest"),(DecisionTreeClassifier(random_state=0),"Decision Tree"),
        (XGBClassifier(),"XGBOOST"),(svm.SVC(kernel='linear',probability=True),"SVM Linear"),
        (CatBoostClassifier(iterations=5,learning_rate=0.1),"CAT Boost")):
    print('=' * 80)
    print(name)

    results_CFS.append(training(clf,X_train_cfs,X_test_cfs,y_cfs,y_test_cfs,"CFS"))
    


# In[ ]:


results_total= []

for clf, name in (
        (GaussianNB() ,"Naive Bayes"),
        (KNeighborsClassifier(n_neighbors = 7),"KNN"),
        (OneVsRestClassifier(svm.SVC(probability=True)),"One vs Rest SVM "),
        (RandomForestClassifier(), "Random forest"),(DecisionTreeClassifier(random_state=0),"Decision Tree"),
        (XGBClassifier(),"XGBOOST"),
        (CatBoostClassifier(iterations=5,learning_rate=0.1),"CAT Boost")):
    print('=' * 80)
    print(name)
    results_total.append(training(clf,X_train,X_test,y,y_test,"Total"))
    


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
indices = np.arange(len(results_total))

results = [[x[i] for x in results_total] for i in range(7)]

clf_names, score,f1score, auc,fpr1,precision,recall = results
plt.figure(figsize=(15, 10))
plt.title("Evaluation of CFS")
plt.barh(indices+.3, f1score, .1, label="f1_score", color='red')
plt.barh(indices+.6, recall, .1, label="Recall", color='blue')
plt.barh(indices+.8, auc, .1, label="AUC", color='c')
plt.barh(indices+.10, score, .1, label="Accuracy", color='darkorange')
plt.barh(indices+.12, precision, .1, label="precision", color='c')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)

plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
  if c == '<catboost.core.CatBoostClassifier object at 0x7fbf12ccd048>':
    plt.text(-.4, i, 'catboost')
  else:

    plt.text(-.4, i, c)

plt.show()
dic={}
auc_dic={}
f1_dic={}
rec={}
prec={}
fpr_={}
for key in clf_names: 
    for value in score: 
        dic[key] = value 
        score.remove(value) 
        break

for key in clf_names: 
    for value in f1score: 
        f1_dic[key] = value 
        f1score.remove(value) 
        break

for key in clf_names: 
    for value in auc: 
        auc_dic[key] = value 
        auc.remove(value) 
        break
for key in clf_names: 
    for value in recall: 
        rec[key] = value 
        recall.remove(value) 
        break

for key in clf_names: 
    for value in precision: 
        prec[key] = value 
        precision.remove(value) 
        break
for key in clf_names: 
    for value in fpr1: 
        fpr_[key] = value 
        fpr1.remove(value) 
        break
print("accuracy for Total",)        
print(dic)
print("="*300)
print(" auc score ",)
print(auc_dic)
print("="*300)
print("F1 score ",)
print(f1_dic)
print("="*300)
print("Recall ",)
print(rec)
print("="*300)
print("Precision ",)
print(prec)
print("="*300)
print("FPR")
print(fpr_)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
indices = np.arange(len(results_CFS))

results = [[x[i] for x in results_CFS] for i in range(7)]

clf_names, score,f1score, auc,fpr2,precision,recall = results
plt.figure(figsize=(15, 10))
plt.title("Evaluation of CFS")
plt.barh(indices+.3, f1score, .1, label="f1_score", color='red')
plt.barh(indices+.6, recall, .1, label="Recall", color='blue')
plt.barh(indices+.8, auc, .1, label="AUC", color='c')
plt.barh(indices+.10, score, .1, label="Accuracy", color='darkorange')
plt.barh(indices+.12, precision, .1, label="precision", color='c')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)

plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
  if c == '<catboost.core.CatBoostClassifier object at 0x7fbf12ccd048>':
    plt.text(-.4, i, 'catboost')
  else:

    plt.text(-.4, i, c)

plt.show()
dic={}
auc_dic={}
f1_dic={}
rec={}
prec={}
fpr_={}
for key in clf_names: 
    for value in score: 
        dic[key] = value 
        score.remove(value) 
        break

for key in clf_names: 
    for value in f1score: 
        f1_dic[key] = value 
        f1score.remove(value) 
        break

for key in clf_names: 
    for value in auc: 
        auc_dic[key] = value 
        auc.remove(value) 
        break
for key in clf_names: 
    for value in recall: 
        rec[key] = value 
        recall.remove(value) 
        break

for key in clf_names: 
    for value in precision: 
        prec[key] = value 
        precision.remove(value) 
        break
for key in clf_names: 
    for value in fpr2: 
        fpr_[key] = value 
        fpr2.remove(value) 
        break
print("accuracy for CFS",)        
print(dic)
print("="*300)
print(" auc score ",)
print(auc_dic)
print("="*300)
print("F1 score ",)
print(f1_dic)
print("="*300)
print("Recall ",)
print(rec)
print("="*300)
print("Precision ",)
print(prec)
print("="*300)
print("FPR")
print(fpr_)


# # Lets try other feature selection technique: Information Gain

# In[11]:


res = mutual_info_classif(X, y, random_state=0)
print('MI for DOS',res)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,15))
feature_importance=pd.Series(res,X.columns[0:len(X.columns)])
feature_importance.plot(kind='barh',color='green')
plt.title("MI for DOS")

plt.show()


# In[12]:


X_=X.drop(['Land','wrong_fragment','Urgent packets','hot','num_failed_logins','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','srv_rerror_rate','Duration'],axis=1)


# In[13]:


X_test_=X_test.drop(['Land','wrong_fragment','Urgent packets','hot','num_failed_logins','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','srv_rerror_rate','Duration'],axis=1)


# In[ ]:


sc = StandardScaler()
X_train_mi = sc.fit_transform(X_)
X_test_mi = sc.transform(X_test_)


# In[ ]:


results_mi= []

for clf, name in (
        (GaussianNB() ,"Naive Bayes"),
        (KNeighborsClassifier(n_neighbors = 7),"KNN"),
        (OneVsRestClassifier(svm.SVC(probability=True)),"One vs Rest SVM "),
        (RandomForestClassifier(), "Random forest"),(DecisionTreeClassifier(random_state=0),"Decision Tree"),
        (XGBClassifier(),"XGBOOST"),
        (CatBoostClassifier(iterations=5,learning_rate=0.1),"CAT Boost")):
    print('=' * 80)
    print(name)
    results_mi.append(training(clf,X_train_mi,X_test_mi,y,y_test,"Mutual Info"))
    


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
indices = np.arange(len(results_mi))

results = [[x[i] for x in results_mi] for i in range(7)]

clf_names, score,f1score, auc,fpr2,precision,recall = results
plt.figure(figsize=(15, 10))
plt.title("Evaluation of MI")
plt.barh(indices+.3, f1score, .1, label="f1_score", color='red')
plt.barh(indices+.6, recall, .1, label="Recall", color='blue')
plt.barh(indices+.8, auc, .1, label="AUC", color='c')
plt.barh(indices+.10, score, .1, label="Accuracy", color='darkorange')
plt.barh(indices+.12, precision, .1, label="precision", color='c')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)

plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
  if c == '<catboost.core.CatBoostClassifier object at 0x7fbf12ccd048>':
    plt.text(-.4, i, 'catboost')
  else:

    plt.text(-.4, i, c)

plt.show()
dic={}
auc_dic={}
f1_dic={}
rec={}
prec={}
fpr_={}
for key in clf_names: 
    for value in score: 
        dic[key] = value 
        score.remove(value) 
        break

for key in clf_names: 
    for value in f1score: 
        f1_dic[key] = value 
        f1score.remove(value) 
        break

for key in clf_names: 
    for value in auc: 
        auc_dic[key] = value 
        auc.remove(value) 
        break
for key in clf_names: 
    for value in recall: 
        rec[key] = value 
        recall.remove(value) 
        break

for key in clf_names: 
    for value in precision: 
        prec[key] = value 
        precision.remove(value) 
        break
for key in clf_names: 
    for value in fpr2: 
        fpr_[key] = value 
        fpr2.remove(value) 
        break
print("accuracy for MI",)        
print(dic)
print("="*300)
print(" auc score ",)
print(auc_dic)
print("="*300)
print("F1 score ",)
print(f1_dic)
print("="*300)
print("Recall ",)
print(rec)
print("="*300)
print("Precision ",)
print(prec)
print("="*300)
print("FPR")
print(fpr_)


# ## Tabulating my results

# In[ ]:


from tabulate import tabulate
print("="*50)
print("|Tabulated Result|")
print("="*50)
data = [[1, 'Gaussian NB', 0.44,0.82,0.44],
[2, 'K Nearest Neighbour', 0.77,0.81,0.77],
[3, 'OnevsRest Classifier', 0.78,0.88,0.78],
[4,'Random Forest', 0.76,0.93,0.76],
[5,'Decision Tree', 0.75,0.75,0.75],
[6,'XGB Classifier', 0.78,0.94,0.78],
[7,'Catboost Classifier', 0.82,0.93,0.82],

] 
print("Report on Total Data")
print("="*80)
print (tabulate(data, headers=["S.No", "Classifier", "Accuracy","AUC"," F1 Score"],tablefmt="psql"))


data_1 = [[1, 'Gaussian NB', 0.57,0.90,0.57],
[2, 'K Nearest Neighbour', 0.78,0.82,0.78],
[3, 'OnevsRest Classifier', 0.79,0.87,0.79],
[4,'Random Forest', 0.78,0.93,0.78],
[5,'Decision Tree', 0.77,0.77,0.77],
[6,'XGB Classifier', 0.80,0.94,0.80],
[7,'Catboost Classifier', 0.76,0.93,0.76]] 
print("Correlation based feature selection")
print("="*80)
print (tabulate(data_1, headers=["S.No", "Classifier", "Accuracy","AUC","F1 Score"],tablefmt="psql"))




data_2 = [[1, 'Gaussian NB', 0.71,0.87,0.71],
[2, 'K Nearest Neighbour', 0.79,0.82,0.79],
[3, 'OnevsRest Classifier', 0.79,0.88,0.79],
[4,'Random Forest', 0.77,0.94,0.77],
[5,'Decision Tree', 0.76,0.76,0.76],
[6,'XGB Classifier', 0.79,0.94,0.79],
[7,'Catboost Classifier', 0.79,0.92,0.79]] 
print("Mutual Information based feature selection")
print("="*80)
print (tabulate(data_2, headers=["S.No", "Classifier", "Accuracy","AUC","F1 Score"],tablefmt="psql"))


# ## MI is behaving best hence we will go with MI features.

# ## Conclusion:
# 

# ## From the above we can conclude the following:
# 1. Mutual Information based feature selection is giving better F1 score and reduced False positive rate. Next we will tune our models on the updated dataset along with removing outliers and evaluate our performance.

# # Custom classification

# # Lets try the same after removing outliers.

# In[40]:


Xnew= pd.concat([X_,y],axis=1)


# In[41]:


Xnew_test= pd.concat([X_test_,y_test],axis=1)


# In[42]:




Xnew= pd.concat([X_,y],axis=1)
Xnew_test= pd.concat([X_test_,y_test],axis=1)
Q1 = Xnew.quantile(0.05)
Q3 = Xnew.quantile(0.95)
IQR = Q3 - Q1

new_df_ = Xnew[~((Xnew < (Q1 - 1.5 * IQR)) |(Xnew > (Q3 + 1.5 * IQR))).any(axis=1)]

Q1 = Xnew_test.quantile(0.05)
Q3 = Xnew_test.quantile(0.95)
IQR = Q3 - Q1

new_df_test_ = Xnew_test[~((Xnew_test < (Q1 - 1.5 * IQR)) |(Xnew_test > (Q3 + 1.5 * IQR))).any(axis=1)]

new_df_ = new_df_.reset_index()
del new_df_['index']
new_df_test_ = new_df_test_.reset_index()
del new_df_test_['index']


# In[43]:


y_oltr=new_df_['attack_type']
X1=new_df_.drop(['attack_type'],axis=1)
y_test_oltr=new_df_test_['attack_type']
X_test1=new_df_test_.drop(['attack_type'],axis=1)
sc = StandardScaler()
X_train_otlr = sc.fit_transform(X1)
X_test_otlr = sc.transform(X_test1)


# ## Lets Tune our models

# **Logistic Regression**

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
model1=LogisticRegression( class_weight='balanced', penalty='l2',solver='newton-cg',C=100,random_state=0,max_iter=1200000)
model1.fit(X_train_otlr, y_oltr)
y_pred_1 = model1.predict(X_test_otlr)
cm = confusion_matrix(y_test_oltr, y_pred_1)
lr_acc_score = accuracy_score(y_test_oltr, y_pred_1)
FPR,prec,rec= falseposrate(cm,y_test_oltr,y_pred_1)

print("\n")
print('-------------------------')
print('| Logistic Regression |')
print('-------------------------')
print("Accuracy  after removing outliers:",lr_acc_score*100,'\n')
print("="*50)
print("F1 score", f1_score(y_test_oltr,y_pred_1,average='micro'))
print("\n")
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
classification_report = metrics.classification_report(y_test_oltr, y_pred_1)
print(classification_report)
print('-------------------------')
print("|Precision|")
print(prec)
print('-------------------------')


print("|False positive rate|")
print('-------------------------')
print(FPR)
print('-------------------------')
pd.crosstab(y_test_oltr, y_pred_1, rownames=['Actual attacks'], colnames=['Predicted attacks'])
fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(model1, X_test_otlr, y_test_oltr,ax=ax,cmap=plt.cm.Blues)
plt.show()


# **Random Forest**

# In[23]:


model2= RandomForestClassifier(criterion = "gini",class_weight='balanced',n_estimators=500, max_depth = 12,max_features = "log2", n_jobs = -1, random_state = 0)
model2.fit(X_train_otlr, y_oltr)
y_pred2 = model2.predict(X_test_otlr)
cm = confusion_matrix(y_test_oltr, y_pred2)
lr_acc_score = accuracy_score(y_test_oltr, y_pred2)
FPR,prec,rec= falseposrate(cm,y_test_oltr,y_pred2)

print("\n")
print('-------------------------')
print('| Random Forest |')
print('-------------------------')
print("Accuracy of Random Forest after removing outliers:",lr_acc_score*100,'\n')
print("="*50)
print("F1 score", f1_score(y_test_oltr,y_pred2,average='micro'))
print("\n")
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
classification_report = metrics.classification_report(y_test_oltr, y_pred2)
print(classification_report)
print('-------------------------')
print("|Precision|")
print(prec)
print('-------------------------')


print("|False positive rate|")
print('-------------------------')
print(FPR)
print('-------------------------')

fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(model2, X_test_otlr, y_test_oltr,ax=ax,cmap=plt.cm.Blues)
plt.show()
pd.crosstab(y_test_oltr, y_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])


# **Decision Tree**

# In[28]:


model3= DecisionTreeClassifier(criterion = "gini",class_weight='balanced', max_depth = 10,  random_state = 0)
model3.fit(X_train_otlr, y_oltr)
y_pred_3 = model3.predict(X_test_otlr)
cm = confusion_matrix(y_test_oltr, y_pred_3)
lr_acc_score = accuracy_score(y_test_oltr, y_pred_3)
FPR,prec,rec= falseposrate(cm,y_test_oltr,y_pred_3)

print("\n")
print('-------------------------')
print('| Decision Tree |')
print('-------------------------')
print("Accuracy of Decision Tree after removing outliers:",lr_acc_score*100,'\n')
print("="*50)
print("F1 score", f1_score(y_test_oltr,y_pred_3,average='micro'))
print("\n")
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
classification_report = metrics.classification_report(y_test_oltr, y_pred_3)
print(classification_report)
print('-------------------------')
print("|Precision|")
print(prec)
print('-------------------------')


print("|False positive rate|")
print('-------------------------')
print(FPR)
print('-------------------------')

fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(model3, X_test_otlr, y_test_oltr,ax=ax,cmap=plt.cm.Blues)
plt.show()
pd.crosstab(y_test_oltr, y_pred_3, rownames=['Actual attacks'], colnames=['Predicted attacks'])


# **Linear Support Vector Machine**

# In[29]:


model4= svm.SVC(C = 10, gamma = 'scale', kernel = "linear",class_weight='balanced',probability=True)
model4.fit(X_train_otlr, y_oltr)
y_pred_4 = model4.predict(X_test_otlr)
cm = confusion_matrix(y_test_oltr, y_pred_4)
lr_acc_score = accuracy_score(y_test_oltr, y_pred_4)
FPR,prec,rec= falseposrate(cm,y_test_oltr,y_pred_4)

print("\n")
print('-------------------------')
print('| Support vector Machine |')
print('-------------------------')
print("Accuracy  on SVM after removing outliers:",lr_acc_score*100,'\n')
print("="*50)
print("F1 score", f1_score(y_test_oltr,y_pred_4,average='micro'))
print("\n")
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
classification_report = metrics.classification_report(y_test_oltr, y_pred_4)
print(classification_report)
print('-------------------------')
print("|Precision|")
print(prec)
print('-------------------------')


print("|False positive rate|")
print('-------------------------')
print(FPR)
print('-------------------------')

fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(model4, X_test_otlr, y_test_oltr,ax=ax,cmap=plt.cm.Blues)
plt.show()
pd.crosstab(y_test_oltr, y_pred_4, rownames=['Actual attacks'], colnames=['Predicted attacks'])


# **Catboost Classifier**

# In[36]:


model5= CatBoostClassifier(iterations=100,learning_rate=0.01,depth=8)
model5.fit(X_train_otlr, y_oltr)
y_pred_5 = model5.predict(X_test_otlr)
cm = confusion_matrix(y_test_oltr, y_pred_5)
lr_acc_score = accuracy_score(y_test_oltr, y_pred_5)
FPR,prec,rec= falseposrate(cm,y_test_oltr,y_pred_5)

print("\n")
print('-------------------------')
print('| CatBoost Classifier |')
print('-------------------------')
print("Accuracy of Catboost Classifier after removing outliers:",lr_acc_score*100,'\n')
print("="*50)
print("F1 score", f1_score(y_test_oltr,y_pred_5,average='micro'))
print("\n")
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
classification_report = metrics.classification_report(y_test_oltr, y_pred_5)
print(classification_report)
print('-------------------------')
print("|Precision|")
print(prec)
print('-------------------------')


print("|False positive rate|")
print('-------------------------')
print(FPR)
print('-------------------------')
fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(model5, X_test_otlr, y_test_oltr,ax=ax,cmap=plt.cm.Blues)
plt.show()


# **K Nearest Neighbour**

# In[38]:


model6= KNeighborsClassifier(n_neighbors = 13,metric='manhattan',weights='distance')
model6.fit(X_train_otlr, y_oltr)
y_pred_6 = model6.predict(X_test_otlr)
cm = confusion_matrix(y_test_oltr, y_pred_6)
lr_acc_score = accuracy_score(y_test_oltr, y_pred_6)
FPR,prec,rec= falseposrate(cm,y_test_oltr,y_pred_6)

print("\n")
print('-------------------------')
print('| KNN |')
print('-------------------------')
print("Accuracy of knn after removing outliers:",lr_acc_score*100,'\n')
print("="*50)
print("F1 score", f1_score(y_test_oltr,y_pred_6,average='micro'))
print("\n")
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
classification_report = metrics.classification_report(y_test_oltr, y_pred_6)
print(classification_report)
print('-------------------------')
print("|Precision|")
print(prec)
print('-------------------------')


print("|False positive rate|")
print('-------------------------')
print(FPR)
print('-------------------------')
fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(model6, X_test_otlr, y_test_oltr,ax=ax,cmap=plt.cm.Blues)
plt.show()
#pd.crosstab(y_test_oltr, y_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
pd.crosstab(y_test_oltr, y_pred_6, rownames=['Actual attacks'], colnames=['Predicted attacks'])


# **XGBoost Classifier**

# In[40]:


model7= XGBClassifier(learning_rate=0.01, n_estimators=50, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
model7.fit(X_train_otlr, y_oltr)
y_pred_7 = model7.predict(X_test_otlr)
cm = confusion_matrix(y_test_oltr, y_pred_7)
lr_acc_score = accuracy_score(y_test_oltr, y_pred_7)
FPR,prec,rec= falseposrate(cm,y_test_oltr,y_pred_7)

print("\n")
print('-------------------------')
print('| XGB Classifier |')
print('-------------------------')
print("Accuracy of XGB Classifier after removing outliers:",lr_acc_score*100,'\n')
print("="*50)
print("F1 score", f1_score(y_test_oltr,y_pred_7,average='micro'))
print("\n")
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
classification_report = metrics.classification_report(y_test_oltr, y_pred_7)
print(classification_report)
print('-------------------------')
print("|Precision|")
print(prec)
print('-------------------------')


print("|False positive rate|")
print('-------------------------')
print(FPR)
print('-------------------------')
fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(model7, X_test_otlr, y_test_oltr,ax=ax,cmap=plt.cm.Blues)
plt.show()

pd.crosstab(y_test_oltr, y_pred_7, rownames=['Actual attacks'], colnames=['Predicted attacks'])


# **MLP Classifier**

# In[41]:


model8= MLPClassifier(activation = "relu", alpha = 0.01, hidden_layer_sizes = (10,10,10),
                            learning_rate = "constant", max_iter = 2000, random_state = 1000)

model8.fit(X_train_otlr, y_oltr)
y_pred_8 = model8.predict(X_test_otlr)
cm = confusion_matrix(y_test_oltr, y_pred_8)
lr_acc_score = accuracy_score(y_test_oltr, y_pred_8)
FPR,prec,rec= falseposrate(cm,y_test_oltr,y_pred_8)

print("\n")
print('-------------------------')
print('| MLP Classifier |')
print('-------------------------')
print("Accuracy of MLP after removing outliers:",lr_acc_score*100,'\n')
print("="*50)
print("F1 score", f1_score(y_test_oltr,y_pred_8,average='micro'))
print("\n")
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
classification_report = metrics.classification_report(y_test_oltr, y_pred_8)
print(classification_report)
print('-------------------------')
print("|Precision|")
print(prec)
print('-------------------------')


print("|False positive rate|")
print('-------------------------')
print(FPR)
print('-------------------------')
fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(model8, X_test_otlr, y_test_oltr,ax=ax,cmap=plt.cm.Blues)
plt.show()
pd.crosstab(y_test_oltr, y_pred_8, rownames=['Actual attacks'], colnames=['Predicted attacks'])


# In[42]:


from mlxtend.classifier import StackingCVClassifier
from mlxtend.classifier import StackingClassifier


# In[43]:


encoder=LabelEncoder()
y_oltr=encoder.fit_transform(y_oltr)
y_test_oltr=encoder.transform(y_test_oltr)


# In[ ]:


params = {"meta_classifier__kernel": ["linear", "rbf", "poly"],
          "meta_classifier__C": [1, 2],
          "meta_classifier__degree": [3, 4, 5],
          "meta_classifier__probability": [True]}


# **Stacking classifier**

# In[ ]:


sclf = StackingCVClassifier(classifiers=[model1, model2,model3,model4,model6,model8], cv=5,
                            meta_classifier = svm.SVC(probability = True))

grid = GridSearchCV(estimator = sclf, 
                        param_grid = params, 
                        cv = 5,
                        verbose = 0,
                        n_jobs = -1)
sclf.fit(X_train_otlr, y_oltr)
y_pred = sclf.predict(X_test_otlr)
cm = confusion_matrix(y_test_oltr, y_pred)
lr_acc_score = accuracy_score(y_test_oltr, y_pred)
FPR,prec,rec= falseposrate(cm,y_test_oltr,y_pred)

print("\n")
print("Accuracy of StackingCVClassifier after removing outliers:",lr_acc_score*100,'\n')
print("="*50)
print("F1 score", f1_score(y_test_oltr,y_pred,average='micro'))
print("\n")
print('-------------------------')
print('| Classifiction Report |')
print('-------------------------')
classification_report = metrics.classification_report(y_test_oltr, y_pred)
print(classification_report)
print('-------------------------')
print("|Precision|")
print(prec)
print('-------------------------')


print("|False positive rate|")
print('-------------------------')
print(FPR)
print('-------------------------')
pd.crosstab(y_test_oltr, y_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])


# # From all the above models we can conclude that Logistic Regression Model is working best on this data and giving us best accuracy. Hence we will save our model1(Logistic Regression model) and use it for prediction.

# In[24]:


with open('finalized_model1.pkl', 'wb') as f:
    pickle.dump(model1, f)


# In[25]:


with open('finalized_model1.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)


# In[26]:



clf_loaded


# In[46]:


a=clf_loaded.predict(X_test_otlr)
f1_score(y_test_oltr,a,average='micro')


# In[50]:


clf_loaded.predict(np.array([-0.25723209, -0.71023544,  0.78868869, 32.05474804, -0.39303381,
       -0.81398572, -0.80657493, -0.43554944, -0.68497085, -0.68282139,
       -0.35566491,  0.79665795, -0.45676949, -0.35270264, -0.55700738,
       -0.29647643,  0.20716574, -0.18570039,  1.78720069,  0.07882338,
       -0.68613097, -0.67511767, -0.37185359, -0.3598238 ,  0.66167171]).reshape(1,-1))


# In[47]:


a


# In[49]:


X_test_otlr[1,:]


# In[38]:


X_test_.iloc[0]


# In[54]:


q=sc.transform(X_test_)


# In[56]:


q[1,:]


# In[ ]:




