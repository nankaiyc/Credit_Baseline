import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# In[2]:
train_bank = pd.read_csv('train_public.csv')
train_internet = pd.read_csv('train_internet.csv')
test = pd.read_csv('test_public.csv')

### 数据预处理
# In[3]:
train_internet.rename(columns={'is_default': 'isDefault'}, inplace=True)
common_cols = []
for col in train_bank.columns:
    if col in train_internet.columns:
        common_cols.append(col)
    else: continue

# In[4]:
# In[5]:
train_bank_left = list(set(list(train_bank.columns)) - set(common_cols))
train_internet_left = list(set(list(train_internet.columns)) - set(common_cols))
# In[6]:

# In[7]:
train1_data = train_internet[common_cols]
train2_data = train_bank[common_cols]
test_data = test[common_cols[:-1]]

# In[8]:
import datetime
# 日期类型：issueDate，earliesCreditLine
# 转换为pandas中的日期类型
train1_data['issue_date'] = pd.to_datetime(train1_data['issue_date'])
# 提取多尺度特征
train1_data['issue_date_y'] = train1_data['issue_date'].dt.year
train1_data['issue_date_m'] = train1_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
train1_data['issue_date_diff'] = train1_data['issue_date'].apply(lambda x: x-base_time).dt.days
train1_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
train1_data.drop('issue_date', axis = 1, inplace = True)

# In[9]:
# 日期类型：issueDate，earliesCreditLine
# 转换为pandas中的日期类型
train2_data['issue_date'] = pd.to_datetime(train2_data['issue_date'])
# 提取多尺度特征
train2_data['issue_date_y'] = train2_data['issue_date'].dt.year
train2_data['issue_date_m'] = train2_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
train2_data['issue_date_diff'] = train2_data['issue_date'].apply(lambda x: x-base_time).dt.days
train2_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
train2_data.drop('issue_date', axis = 1, inplace = True)

# In[10]:
employer_type = train1_data['employer_type'].value_counts().index
industry = train1_data['industry'].value_counts().index

# In[11]:
emp_type_dict = dict(zip(employer_type, [0,1,2,3,4,5]))
industry_dict = dict(zip(industry, [i for i in range(15)]))

# In[12]:
train1_data['work_year'].fillna('10+ years', inplace=True)
train2_data['work_year'].fillna('10+ years', inplace=True)

work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
train1_data['work_year']  = train1_data['work_year'].map(work_year_map)
train2_data['work_year']  = train2_data['work_year'].map(work_year_map)

train1_data['class'] = train1_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
train2_data['class'] = train2_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

train1_data['employer_type'] = train1_data['employer_type'].map(emp_type_dict)
train2_data['employer_type'] = train2_data['employer_type'].map(emp_type_dict)

train1_data['industry'] = train1_data['industry'].map(industry_dict)
train2_data['industry'] = train2_data['industry'].map(industry_dict)

# In[13]:
# 日期类型：issueDate，earliesCreditLine
#train[cat_features]
# 转换为pandas中的日期类型
test_data['issue_date'] = pd.to_datetime(test_data['issue_date'])
# 提取多尺度特征
test_data['issue_date_y'] = test_data['issue_date'].dt.year
test_data['issue_date_m'] = test_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
test_data['issue_date_diff'] = test_data['issue_date'].apply(lambda x: x-base_time).dt.days
test_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
test_data.drop('issue_date', axis = 1, inplace = True)
test_data['work_year'].fillna('10+ years', inplace=True)

work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
test_data['work_year']  = test_data['work_year'].map(work_year_map)
test_data['class'] = test_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
test_data['employer_type'] = test_data['employer_type'].map(emp_type_dict)
test_data['industry'] = test_data['industry'].map(industry_dict)

# In[13]:combine the data
X_train1 = train1_data.drop(['isDefault','earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)
y_train1 = train1_data['isDefault']
X_train2 = train2_data.drop(['isDefault','earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)
y_train2 = train2_data['isDefault']

X_train_all = pd.concat([X_train1, X_train2]) ##总训练集
y_train_all = pd.concat([y_train1, y_train2]) ##总训练集的分类结果

#X_test = test_data.drop(['earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False) ##总测试集

# In[14]:split the data
def trainTestSplit(X, Y, train_num_of_X):
    '''
    This function can split the data into desire num for test and train by random.
    Variables Describe:
    X: Datafram without label
    Y: Data labels
    train_num_of_X: numbers of train set
    '''
    X_num = X.shape[0]
    test_index = list(range(X_num))
    train_index = []
    train_num = train_num_of_X
    for i in range(train_num):
        randomIndex = int(np.random.uniform(0, len(test_index)))  # Choose train set by random
        train_index.append(test_index[randomIndex])
        del test_index[randomIndex]
    # Control the label consistency
    train = X.iloc[train_index]
    label_train = Y.iloc[train_index]
    test = X.iloc[test_index]
    label_test = Y.iloc[test_index]
    return train, test, label_train, label_test

# In[15]:
X_train, X_test, y_train, y_test = trainTestSplit(X_train_all, y_train_all, int(X_train_all.shape[0] / 2))


# print(X_train_list[0])
# print(X_train_list[1])
# print(X_train_list[0].shape)
# print(X_train_list[1].shape)

# In[16]:
'''
def standardize(train, test, lable):
    mean_px = train.mean().astype(np.float32)
    std_px = train.std().astype(np.float32)
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    X_train_NN = (train - mean_px) / std_px
    X_test_NN = (X_test - mean_px) / std_px
    X_train_NN = (train.values).astype('float32')  # all pixel values
    y_train_NN = lable.astype('int32')
    X_test_NN = (X_test.values).astype('float32')  # all pixel values
    return X_train_NN, y_train_NN, X_test_NN


for i in range(len(X_train_list)):
    X_train_list[i], X_test_list[i], y_train_list[i]  = standardize(X_train_list[i], X_test_list[i], y_train_list[i])
print(X_train_list[0])
print(y_train_list[0])
print(X_test_list[0])
'''
