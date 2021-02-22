#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))

project_data_train = pd.read_csv('KDDTrain+.txt')

#renaming the columns

df_train = project_data_train.rename(columns={"0":"Duration","tcp":"protocol_type","ftp_data":"service","SF":"flag","491":"src_bytes",
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

sc=StandardScaler()
from sklearn import preprocessing 



@app.route('/')
def home():
    if request.method == "POST": 
       return render_template("index.html") 
#@app.route('/')
#def home():
    #return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
   
    print(int_features[0])
    print(int_features[1])
    print(int_features[2])
    label_encoder1 = preprocessing.LabelEncoder() 

    df_train['protocol_type']= label_encoder1.fit_transform(df_train['protocol_type']) 
    a=label_encoder1.classes_
    
    label_encoder2 = preprocessing.LabelEncoder() 

    df_train['service']= label_encoder2.fit_transform(df_train['service']) 
    b=label_encoder2.classes_
    
    label_encoder3 = preprocessing.LabelEncoder() 

    df_train['flag']= label_encoder3.fit_transform(df_train['flag']) 
    c=label_encoder3.classes_
    for i in range(len(a)):
        if a[i]==int_features[0]:
            int_features[0]=i
    for j in range(len(b)):
        if b[j]==int_features[1]:
            int_features[1]=j
    for k in range(len(c)):
        if c[k]==int_features[2]:
            int_features[2]=k
    y=df_train['attack_type']
    X=df_train.drop(['attack_type'],axis=1)
    X_=X.drop(['Land','wrong_fragment','Urgent packets','hot','num_failed_logins','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','srv_rerror_rate','Duration'],axis=1)
    Xnew= pd.concat([X_,y],axis=1)
    #Xnew_test= pd.concat([X_test_,y_test],axis=1)
    Q1 = Xnew.quantile(0.05)
    Q3 = Xnew.quantile(0.95)
    IQR = Q3 - Q1

    new_df_ = Xnew[~((Xnew < (Q1 - 1.5 * IQR)) |(Xnew > (Q3 + 1.5 * IQR))).any(axis=1)]

    #Q1 = Xnew_test.quantile(0.05)
    #Q3 = Xnew_test.quantile(0.95)
    #IQR = Q3 - Q1

    #new_df_test_ = Xnew_test[~((Xnew_test < (Q1 - 1.5 * IQR)) |(Xnew_test > (Q3 + 1.5 * IQR))).any(axis=1)]

    new_df_ = new_df_.reset_index()
    del new_df_['index']
    #new_df_test_ = new_df_test_.reset_index()
    #del new_df_test_['index']
    y_oltr=new_df_['attack_type']
    X1=new_df_.drop(['attack_type'],axis=1)
    #y_test_oltr=new_df_test_['attack_type']
    #X_test1=new_df_test_.drop(['attack_type'],axis=1)
    sc = StandardScaler()
    X_train_otlr = sc.fit_transform(X1)
    
    
    
    final_features = np.array(int_features)
    final_features = final_features.reshape(1,-1)
    final_features=sc.transform(final_features)
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text='Traffic is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




