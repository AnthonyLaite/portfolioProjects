#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)  # Adjusts the configuration of the plots we will create

# Read in data
df = pd.read_csv('/Users/anthonylaite/Desktop/projects/WorldExpenditures.csv')

# Drop the unnamed column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 

# Display the entire DataFrame
df


# In[19]:


for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}'.format(col, pct_missing))



# In[26]:


df.dtypes


# In[32]:


df['Country'].drop_duplicates().sort_values(ascending=False)


# In[33]:


import matplotlib.pyplot as plt

# Scatter plot to visualize the relationship between Expenditure and GDP (%)
plt.scatter(x=df['Expenditure(million USD)'], y=df['GDP(%)'])
plt.title('Scatter Plot: Expenditure vs. GDP (%)')
plt.xlabel('Expenditure (million USD)')
plt.ylabel('GDP (%)')
plt.show()


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt


sns.regplot(x='Expenditure(million USD)', y='GDP(%)', data=df, scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
plt.title('Regression Plot: Expenditure vs. GDP (%)')
plt.show()


# In[35]:


import pandas as pd

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix_pearson = df[numeric_columns].corr()

print(correlation_matrix_pearson)


# In[4]:


# Import libraries
import pandas as pd

# Read in data
df = pd.read_csv('/Users/anthonylaite/Desktop/projects/WorldExpenditures.csv')

# Drop the unnamed column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Replace 'Country' and 'Sector' columns with numerical values
df['Country'] = pd.Categorical(df['Country'])
df['Country'] = df['Country'].cat.codes

df['Sector'] = pd.Categorical(df['Sector'])
df['Sector'] = df['Sector'].cat.codes

# Display the modified DataFrame
(df)




# In[36]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix_pearson = df[numeric_columns].corr()

# Display the correlation matrix
print(correlation_matrix_pearson)

# Create a heatmap with a custom color map and title
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.heatmap(correlation_matrix_pearson, annot=True, cmap="RdBu_r", fmt=".2f")
plt.title("Pearson Correlation Test", fontsize=16)  # Add the title

plt.show()


# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

categorical_columns = ['Country', 'Sector']
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
selected_columns = categorical_columns + list(numeric_columns)

subset_df = df[selected_columns]
correlation_matrix_pearson = subset_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_pearson, annot=True, cmap="RdBu_r", fmt=".2f", xticklabels=selected_columns, yticklabels=selected_columns)
plt.title("Pearson Correlation Test (including Country and Sector)", fontsize=16)

plt.show()



# In[15]:


import pandas as pd

correlation_mat = df.corr()

(correlation_mat)


# In[16]:


import pandas as pd

correlation_mat = df.corr()

# Unstack the correlation matrix and sort the pairs
corr_pairs = correlation_mat.unstack().sort_values(ascending=False)

print(corr_pairs)


# In[18]:


import pandas as pd

correlation_mat = df.corr()

mask = (correlation_mat != 1)

sorted_pairs = (correlation_mat.where(mask)
                .unstack()
                .sort_values(ascending=False)
                .drop_duplicates())

filtered_pairs = sorted_pairs[(sorted_pairs > 0.29) & (sorted_pairs < 1.0)]
print(filtered_pairs)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




