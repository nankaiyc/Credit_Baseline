from Data_Processing import *

# ## 模型使用
# 1) LightGBM
# In[16]:
import lightgbm
from sklearn import metrics
# 利用Internet数据预训练模型1
clf_ex=lightgbm.LGBMRegressor(n_estimators = 200)
clf_ex.fit(X = X_train, y = y_train)
clf_ex.booster_.save_model('LGBMmode.txt')
pred = clf_ex.predict(X_test)

# In[17]:
# submission
submission = pd.DataFrame({'id':test['loan_id'], 'is_default':pred})
submission.to_csv('submission.csv', index = None)
