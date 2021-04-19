# %% [markdown]
# Table of Content
# <br><a href="#linearity">1. Linearity</a>
# <br><a href="#mean">2. Mean of Residuals</a>
# <br><a href="#homo">3. Check for Homoscedasticity</a>
# <br><a href="#normal">4. Check for Normality of error terms/residuals</a>
# <br><a href="#auto">5. No autocorrelation of residuals</a>
# <br><a href="#multico">6. No perfect multicollinearity</a>
# <br><a href="#other">7. Other Models for comparison</a>
# 

# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt

# %% [code]
ad_data = pd.read_csv('../input/Advertising.csv',index_col='Unnamed: 0')

# %% [code]
ad_data.info()

# %% [code]
ad_data.describe()

# %% [code]
p = sns.pairplot(ad_data)

# %% [markdown]
# #  Assumptions for Linear Regression

# %% [markdown]
# ## <a id="linearity">1. Linearity</a>
# 

# %% [code]
p = sns.pairplot(ad_data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7)

# %% [code]
x = ad_data.drop(["Sales"],axis=1)
y = ad_data.Sales

# %% [code]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)

# %% [code]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,test_size=0.25)

# %% [code]
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_train)

# %% [code]
print("R squared: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))

# %% [markdown]
# ## <a id="mean">2. Mean of Residuals</a>

# %% [code]
residuals = y_train.values-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))

# %% [markdown]
# ### Very close to zero so all good here.

# %% [markdown]
# ## <a id="homo">3. Check for Homoscedasticity</a>

# %% [code]
p = sns.scatterplot(y_pred,residuals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.xlim(0,26)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')

# %% [code]
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuals, X_train)
lzip(name, test)

# %% [code]
from scipy.stats import bartlett
test = bartlett( X_train,residuals)
print(test)

# %% [markdown]
# ## <a id="normal">4. Check for Normality of error terms/residuals</a>

# %% [code]
p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of error terms/residuals')

# %% [markdown]
# ## <a id="auto">5. No autocorrelation of residuals</a>

# %% [code] {"scrolled":true}
plt.figure(figsize=(10,5))
p = sns.lineplot(y_pred,residuals,marker='o',color='blue')
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.xlim(0,26)
p = sns.lineplot([0,26],[0,0],color='red')
p = plt.title('Residuals vs fitted values plot for autocorrelation check')

# %% [code]
from statsmodels.stats import diagnostic as diag
min(diag.acorr_ljungbox(residuals , lags = 40)[1])

# %% [code]
import statsmodels.api as sm

# %% [code]
# autocorrelation
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.show()

# %% [code]
# partial autocorrelation
sm.graphics.tsa.plot_pacf(residuals, lags=40)
plt.show()

# %% [markdown]
# ## <a id="multico">6. No perfect multicollinearity</a>

# %% [code]
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(ad_data.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap