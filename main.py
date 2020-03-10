import pandas as pd 
import statsmodels.api as sm
import statsmodels
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
#ler csv com pandas
data = pd.read_csv("data.csv")
#data.head()

#print(data)

#fig, ax = plt.subplots(figsize=(8,6))
#fig = data.boxplot('DESEM', 'NG', ax=ax, grid=False)
#plt.show()
#data.hist()
#plt.show()

#fig = interaction_plot(data.CRS, data.TP, data.DESEM)
#plt.show()

#subset = np.mean(np.array(data.loc[(data.CRS == 2)]['DESEM']))
#print(subset)

#ajustar modelo de regressão linear
mod = ols('SNR2 ~ NG + MUT+ RST + CRS', data=data).fit()
#mod.summary()
#fazer anova
aov_table = sm.stats.anova_lm(mod)
#printar tabela da anova
print(aov_table)
