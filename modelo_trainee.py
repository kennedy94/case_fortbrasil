import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
 
from unidecode import unidecode
#accented_string = 'Málaga'
#dadosX_sem_acento = unidecode.unidecode(dadosX)
 
 
dados = pd.read_csv("Case_Trainee_Fortbrasil.txt",encoding = "ISO-8859-1")
dados.head()
 
dados.plot()
 
plt.boxplot(dados[['TempoNaFortbrasilEmMeses']])
 
dadosX = dados[['TempoNaFortbrasilEmMeses','DataNascimento','QtdComprasUlt12Meses',
               'QtdComprasUlt6Meses','QtdComprasUlt3Meses','QtdComprasUltMes','Sexo','UF',
               'Cidade','NumeroDeDependentes','EstadoCivilDescricao','Atividade_Emissor',
               'PossuiCartaoAdicional','PossuiEmailCadastrado']]
dadosY1 = dados[['PossuiFaturaPorEmail']]
dadosY2 = dados[['PossuiSMSAlerta']]
 
 
#dadosX['Atividade_Emissor'] = dadosX['Atividade_Emissor'].apply(unidecode)
 
 
dadosX = dadosX.fillna('NaN')
dadosX.head()
for column in dadosX:
    if dadosX[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        dadosX[column] = le.fit_transform(dadosX[column])
 
#modelos de ML
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
 
scores = []
for clf in models:
    scores.append(np.mean
                  (np.array
                   (cross_val_score(clf, dadosX, dadosY1.values.ravel(), cv=10))))
     
 
print(scores)
 
scores = np.array(scores)
 
#pega melhor modelo e treina 
melhor_modelo = np.argmax(scores)
 
 
 
 
 
dadosX['EstadoCivilDescricao'] = preprocessing.LabelEncoder().fit_transform(dadosX['EstadoCivilDescricao'].astype(str))
 
enc = preprocessing.LabelEncoder()
 
print(dadosX['EstadoCivilDescricao'].unique())
 
NovoX = dadosX.apply(preprocessing.LabelEncoder().fit_transform)
 
enc.fit(dadosX[['Atividade_Emissor']])
new_cat_features = enc.transform(dadosX[['Atividade_Emissor']])
 
new_cat_features.head()
 
enc = preprocessing.OneHotEncoder()
 
enc.fit(dadosX)
 
unique_x = list(enc.classes_)
 
 
#PCA para 2 componentes
pca = PCA(n_components='mle')
#ajustando modelo do pca para os dados X
dadosX_pca = pca.fit_transform(dadosX)
dadosX.shape
dadosX_pca.shape
 
clf = LDA()
dadosX_pca = clf.fit_transform(dadosX_pca, dadosY1)
dadosX_pca.shape
 
print(pd.DataFrame(clf.components_,columns=dadosX.columns))
 
 
label_encoder = enc.fit(dadosX)
y = label_encoder.transform(dadosX) 
 
dadosX.head();
#dados.columns()
 
plot = dados['PossuiFaturaPorEmail'].value_counts().plot(kind='pie')
plot.axis("equal")
fig = plot.get_figure()
fig.savefig('porSexo.png', dpi = 250)
 
df = dados[(dados['PossuiEmailCadastrado'] == 1) & (dados['PossuiFaturaPorEmail'] == 1)]
 
df = dados[(dados['PossuiFaturaPorEmail'] == 1)]
df.shape
df.head()