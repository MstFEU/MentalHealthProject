#Importing Library
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

#Importing Data
dataset = pd.read_csv("D:\Artificial Neural Network\MENTAL PROJ\Database\Mental health Data\Mental Health Data.csv")

#Rename the columns
a=list(dataset.columns)
b=['self_employed','no_of_employees','tech_company','role_IT','mental_healthcare_coverage',
  'knowledge_about_mental_healthcare_options_workplace','employer_discussed_mental_health',
  'employer_offer_resources_to_learn_about_mental_health','medical_leave_from_work',
  'comfortable_discussing_with_coworkers','employer_take_mental_health_seriously',
   'knowledge_of_local_online_resources ','productivity_affected_by_mental_health ',
   'percentage_work_time_affected_mental_health','openess_of_family_friends',
  'family_history_mental_illness','mental_health_disorder_past',
   'currently_mental_health_disorder','diagnosed_mental_health_condition',
   'type_of_disorder','treatment_from_professional',
   'while_effective_treatment_mental_health_issue_interferes_work',
  'while_not_effective_treatment_interferes_work','age','gender','country','US state',
  'country work ','US state work','role_in_company','work_remotely','']
for i,j in zip(a,b):
    dataset.rename(columns={i:j},inplace=True)

#Copy the dataset in df1
df1 = dataset.copy()

#Drop the unnecessary colums 
cols = ['role_IT','knowledge_of_local_online_resources ',
        'productivity_affected_by_mental_health ',
        'percentage_work_time_affected_mental_health']
df_red = df1.drop(cols,axis=1)

#Data Cleaning
df_red.no_of_employees.replace(to_replace=['1 to 5', '6 to 25','More than 1000','26-99'],
                                value=['1-5','6-25','>1000','26-100'],inplace=True)

df_red.mental_healthcare_coverage.replace(to_replace=['Not eligible for coverage / N/A'],
                                value='No',inplace=True)

df_red.openess_of_family_friends.replace(to_replace=['Not applicable to me (I do not have a mental illness)']
                                          ,value="I don't know",inplace=True)

med_age = df_red[(df_red['age'] >= 18) | (df_red['age'] <= 75)]['age'].median()
df_red['age'].replace(to_replace = df_red[(df_red['age'] < 18) | (df_red['age'] > 75)]['age'].tolist(), value = med_age, inplace = True)

df_red['gender'].replace(to_replace = ['Male', 'male', 'Male ', 'M', 'm',
       'man', 'Cis male', 'Male.', 'male 9:1 female, roughly', 'Male (cis)', 'Man', 'Sex is male',
       'cis male', 'Malr', 'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
       'mail', 'M|', 'Male/genderqueer', 'male ',
       'Cis Male', 'Male (trans, FtM)',
       'cisdude', 'cis man', 'MALE'], value = 'male', inplace = True)
df_red['gender'].replace(to_replace = ['Female', 'female', 'I identify as female.', 'female ',
       'Female assigned at birth ', 'F', 'Woman', 'fm', 'f', 'Cis female ', 'Transitioned, M2F',
       'Genderfluid (born female)', 'Female or Multi-Gender Femme', 'Female ', 'woman', 'female/woman',
       'Cisgender Female', 'fem', 'Female (props for making this a freeform field, though)',
       ' Female', 'Cis-woman', 'female-bodied; no feelings about gender',
       'AFAB'], value = 'female', inplace = True)
df_red['gender'].replace(to_replace = ['Bigender', 'non-binary', 'Other/Transfeminine',
       'Androgynous', 'Other', 'nb masculine',
       'none of your business', 'genderqueer', 'Human', 'Genderfluid',
       'Enby', 'genderqueer woman', 'mtf', 'Queer', 'Agender', 'Fluid',
       'Nonbinary', 'human', 'Unicorn', 'Genderqueer',
       'Genderflux demi-girl', 'Transgender woman'], value = 'other', inplace = True)

tech_list = []
tech_list.append(df_red[df_red['role_in_company'].str.contains('Back-end')]['role_in_company'].tolist())
tech_list.append(df_red[df_red['role_in_company'].str.contains('Front-end')]['role_in_company'].tolist())
tech_list.append(df_red[df_red['role_in_company'].str.contains('Dev')]['role_in_company'].tolist())
tech_list.append(df_red[df_red['role_in_company'].str.contains('DevOps')]['role_in_company'].tolist())
flat_list = [item for sublist in tech_list for item in sublist]
flat_list = list(dict.fromkeys(flat_list))

df_red['tech_role']=df_red['role_in_company']
df_red['tech_role'].replace(to_replace=flat_list,value=1,inplace=True)
remain_list=df_red['tech_role'].unique()[1:]
df_red['tech_role'].replace(to_replace=remain_list,value=0,inplace=True)

df_red.tech_role.value_counts()
df_red=df_red.drop(['role_in_company'],axis=1)

#Dealing with missing data
df_rej=pd.concat([df_red['type_of_disorder'],df_red['US state'],df_red['US state work']],axis=1) #ทำการนำ 3 คอลัมน์นี้มาต่อหัวกันให้เป็น 1 คอลัมน์
df_red=df_red.drop(['type_of_disorder','US state','US state work'],axis=1) #ดรอปสามคอลัมน์นี้ไปแล้ว

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') #ใช้ SimpleImputer มาแทนที่ค่าที่ nan ด้วยค่าที่พบบ่อยสุดในคอมลัมน์นั้น ๆ (most_frequent)
imp.fit(df_red)
imp_data=pd.DataFrame(data=imp.transform(df_red),columns=df_red.columns)

df_eda=pd.concat([imp_data,df_rej],axis=1)
df_eda.isnull().sum().to_frame() #returns the number of missing values in the data set. 

#Machine Learning with Scikit-Learning
## Dropping unnecessary columns
y = df_eda.diagnosed_mental_health_condition
X = df_eda.drop(['diagnosed_mental_health_condition','treatment_from_professional',
               'while_effective_treatment_mental_health_issue_interferes_work',
               'while_not_effective_treatment_interferes_work','type_of_disorder','US state',
               'country', 'US state work', 'country work '],axis=1) #'country work ' NOT 'country work' นะ คือขี้เกียจไปแก้ละ

X_conv = X.copy()
y_conv = y.copy()

cat_columns=['self_employed', 'no_of_employees', 'tech_company',
       'mental_healthcare_coverage',
       'knowledge_about_mental_healthcare_options_workplace',
       'employer_discussed_mental_health',
       'employer_offer_resources_to_learn_about_mental_health',
       'medical_leave_from_work', 'comfortable_discussing_with_coworkers',
       'employer_take_mental_health_seriously', 'openess_of_family_friends',
       'family_history_mental_illness', 'mental_health_disorder_past',
       'currently_mental_health_disorder', 'age', 'gender', 'work_remotely', 'tech_role']

## Splitting the data
X_train, X_valid, y_train, y_valid = train_test_split(X_conv ,y_conv ,train_size=0.8,test_size=0.2,random_state=0)

##Label Encode the data 
from sklearn.preprocessing import LabelEncoder
label_encode = LabelEncoder() #บางเว็ปใช้ le
label_X_train=X_train.copy()
label_X_valid=X_valid.copy()

for col in cat_columns:
    label_X_train[col] = label_encode.fit_transform(X_train[col])
    label_X_valid[col] = label_X_valid[col].map(lambda s: '<unknown>' if s not in label_encode.classes_ else s)
    label_encode.classes_ = np.append(label_encode.classes_, '<unknown>')
    label_X_valid[col] = label_encode.fit_transform(label_X_valid[col]) #เพิ่ม fit_trandform เข้าไปแทน transform เฉยๆ

label_y_train=label_encode.fit_transform(y_train)
label_y_valid=label_encode.transform(y_valid)

##Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier #predict with improved accuracy 
model=RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model.fit(label_X_train,label_y_train)
preds=model.predict(label_X_valid)
decoded_preds = label_encode.inverse_transform(preds)

#Evaluate an input list
input_list = [[0, 5,	1, 2, 0, 1, 0, 4, 0, 1, 3, 2, 1, 2, 26, 1, 0, 1]]
predicted = new_preds=model.predict(input_list)
decoded_predicted = label_encode.inverse_transform(predicted)
print(decoded_predicted)

#save model to pkl file
import pickle
#pickle_out = open("model.pkl","wb")
#pickle.dump(model, pickle_out)
#loaded_model = pickle.load(open("model.pkl", "rb"))
pickle.dump(model, open('model.pkl', 'wb'))
