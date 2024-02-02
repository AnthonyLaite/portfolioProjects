#!/usr/bin/env python
# coding: utf-8

# In[65]:


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


df.head()





# In[188]:


# Assuming df_numerized is your DataFrame
df_no_short = df_numerized.drop(columns=['sectorshort'])

# Check for missing data in each column
for col in df_no_short.columns:
    pct_missing = np.mean(df_no_short[col].isnull())
    print(f'{col} - {pct_missing * 100:.2f}%')



# In[80]:


# Display data types for columns
print(df.dtypes)


# In[104]:


# Change data types of columns
df['Expenditure(million USD)'] = df['Expenditure(million USD)'].astype('float64')
df['Year'] = df['Year'].astype('int64')






# In[ ]:






# In[189]:


# Assuming df is your DataFrame
df_no_short = df.drop(columns=['sectorshort'])

# Sort the DataFrame by 'Expenditure(million USD)' column
sorted_df = df_no_short.sort_values(by=['Expenditure(million USD)'], ascending=False)

# Display the sorted DataFrame
sorted_df


# In[ ]:





# In[114]:


#drop any duplicate 

df['Country'].drop_duplicates().sort_values(ascending=False)


# In[118]:


import matplotlib.pyplot as plt

# Scatter plot to visualize the relationship between Expenditure and GDP (%)
plt.scatter(x=df['Expenditure(million USD)'], y=df['GDP(%)'])
plt.title('Scatter Plot: Expenditure vs. GDP (%)')
plt.xlabel('Expenditure (million USD)')
plt.ylabel('GDP (%)')
plt.show()


# In[119]:


df.head()
df


# In[128]:


#plot budget vs groos using seaborn 

import seaborn as sns
import matplotlib.pyplot as plt


sns.regplot(x='Expenditure(million USD)', y='GDP(%)', data=df, scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
plt.title('Regression Plot: Expenditure vs. GDP (%)')
plt.show()



# In[134]:


import pandas as pd

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix_pearson = df[numeric_columns].corr()

print(correlation_matrix_pearson)




# In[139]:


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




# In[ ]:






# In[141]:


df_numerized = df 

For col_name in df_numerized.columns 
if(df)numerized[col_name].dtype == 'object'): 
    df_numerized[col_name] = df_numerized[col_name].astype('category')
    df_numerized[col_name = df_numerized[col_name].cat.codes 
    
    df_numerized


# In[ ]:



    




# In[ ]:





# In[ ]:





# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df_numerized is your DataFrame
correlation_matrix_pearson = df_numerized.corr()

# Create a heatmap with a custom color map and title
plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_pearson, annot=True, cmap="RdBu_r", fmt=".2f")
plt.title("Pearson Correlation Test", fontsize=16)  # Add the title

plt.show()



# In[193]:


correlation_mat = df_numerized.drop(columns=['sectorshort']).corr()

corr_pairs = correlation_mat.unstack()

corr_pairs



# In[194]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[195]:


filtered_corr = sorted_pairs[(sorted_pairs > 0.29) & (sorted_pairs < 1.0) & (sorted_pairs.index.get_level_values(0) != 'sectorshort') & (sorted_pairs.index.get_level_values(1) != 'sectorshort')].drop_duplicates()
filtered_corr




# In[ ]:




