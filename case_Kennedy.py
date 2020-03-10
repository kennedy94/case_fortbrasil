import pandas as pd 
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import pylab
import matplotlib as mpl
    
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib_venn import venn2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def Ler_Tratar_Dados():
    #ler csv com pandas
    dados = pd.read_csv("Case_Trainee_Fortbrasil.csv",encoding = "ISO-8859-1")
    print(dados.shape)
    #setar indices do DataFrame como IdConta
    dados.index = dados['IdConta']
    dados.drop('IdConta', axis = 1, inplace = True)
    
    #Tratar Valores faltantes
    dados = dados.fillna('NaN')
    
    #transformar data de nascimento em idade
    import datetime as DT
    df = dados
    now = pd.Timestamp(DT.datetime.now())
    df['DataNascimento'] = pd.to_datetime(df['DataNascimento'], format="%Y-%m-%d") #1
    df['DataNascimento'] = df['DataNascimento'].where(df['DataNascimento'] < now, df['DataNascimento'] -  np.timedelta64(100, 'Y'))   # 2
    dados['DataNascimento'] = (now - df['DataNascimento']).astype('<m8[Y]') # 3
    
    dados = dados.rename(index=str, columns={"DataNascimento": "Idade"})
    
    #remover acentos
    #import unidecode
    dados['Atividade_Emissor'] = dados['Atividade_Emissor'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    #remover anomalias que identifiquei nos dados
    indexNames1 = dados[ (dados.Idade < 18) | (dados.Idade > 100)].index
    indexNames2 = dados[ (dados.PossuiEmailCadastrado == 0) & 
                        (dados.PossuiFaturaPorEmail == 1)].index
    dados.drop(indexNames1, inplace=True)
    dados.drop(indexNames2, inplace=True)
    
    dados = dados.drop_duplicates()#dados.columns.difference(['IdConta']))
    print(dados.shape)
    return dados
    
    

def Plotar_Venn(dados):
    
    sn.set()
    
    i1 = dados[ (dados.PossuiFaturaPorEmail == 1)].index
    i2 = dados[ (dados.PossuiSMSAlerta == 1)].index
    venn2([set(i1),set(i2)], set_labels = ('PossuiFaturaPorEmail', 'PossuiSMSAlerta'))
    plt.savefig('Venn.png', dpi = 250)


def Plotar_Histogramas(dados):
    dados_Email = dados[(dados.PossuiFaturaPorEmail == 1)]
    dados_SMS = dados[(dados.PossuiSMSAlerta == 1)]
    
    print(dados['Atividade_Emissor'].unique())
    
    
    dados['Atividade_Emissor'].plot(kind = 'pie')
    dados_SMS['Idade'].hist()
    
    plot = pd.DataFrame({'PossuiFaturaPorEmail': dados_Email.Idade,
                  'PossuiSMSAlerta':   dados_SMS.Idade}).plot.hist(stacked = True)
    
    plot = dados.groupby('PossuiFaturaPorEmail').get_group(0).Idade.plot.pie()#stacked = True)
    
    plot = pd.DataFrame({'PossuiFaturaPorEmail': dados.groupby('PossuiFaturaPorEmail').get_group(1).Idade,
                  'NaoPossuiFaturaPorEmail':dados.groupby('PossuiFaturaPorEmail').get_group(0).Idade}).plot.hist(stacked = True)
    
    
    plot = pd.DataFrame({'PossuiSMSAlerta': dados.groupby('PossuiSMSAlerta').get_group(1).Idade,
                  'NaoPossuiSMSAlerta':dados.groupby('PossuiSMSAlerta').get_group(0).Idade}).plot.hist(stacked = True)
    
    
    
    plot = pd.DataFrame({'PossuiFaturaPorEmail': dados.groupby('PossuiFaturaPorEmail').get_group(1).TempoNaFortbrasilEmMeses,
                  'NaoPossuiFaturaPorEmail':dados.groupby('PossuiFaturaPorEmail').get_group(0).TempoNaFortbrasilEmMeses}).plot.hist(stacked = True)
    
    
    plot = pd.DataFrame({'PossuiSMSAlerta': dados.groupby('PossuiSMSAlerta').get_group(1).TempoNaFortbrasilEmMeses,
                  'NPossuiSMSAlerta':dados.groupby('PossuiSMSAlerta').get_group(0).TempoNaFortbrasilEmMeses}).plot.hist(stacked = True)
    
    
    
    
    fig = plot.get_figure()
    fig.savefig('porIdade_hist_SMS.png', dpi = 250)



def Plotar_Histogramas_TempoNaFB(dados):
    plot = pd.DataFrame({'PossuiFaturaPorEmail': 
        dados.groupby('PossuiFaturaPorEmail').get_group(1).TempoNaFortbrasilEmMeses,
                  'NaoPossuiFaturaPorEmail':
                      dados.groupby('PossuiFaturaPorEmail').
                      get_group(0).TempoNaFortbrasilEmMeses}).plot.hist(stacked = True, bins = (2,3,4,5,6,7,8,9,10,11,12))
    plt.xticks(np.arange(2,13)+.5,np.arange(2,12))
    fig = plot.get_figure()
    fig.savefig('hist_tempoFB_Fatura.png', dpi = 250)
    
    plot = pd.DataFrame({'PossuiSMSAlerta': 
        dados.groupby('PossuiSMSAlerta').get_group(1).TempoNaFortbrasilEmMeses,
                  'NaoPossuiSMSAlerta':
                      dados.groupby('PossuiSMSAlerta').get_group(0).TempoNaFortbrasilEmMeses}).plot.hist(stacked = True, bins = (2,3,4,5,6,7,8,9,10,11,12))
    plt.xticks(np.arange(2,13)+.5,np.arange(2,12))
    
    fig = plot.get_figure()
    fig.savefig('hist_SMS_Fatura.png', dpi = 250)
    
    
    
def Tabelas_Cidade(dados):
    dados.groupby('PossuiFaturaPorEmail').get_group(1).Cidade.value_counts()[:11]#.plot(kind = 'bar')
    dados.groupby('PossuiFaturaPorEmail').get_group(0).Cidade.value_counts()[:11]#.plot(kind = 'bar')
    dados.groupby('PossuiSMSAlerta').get_group(1).Cidade.value_counts()[:11]
    dados.groupby('PossuiSMSAlerta').get_group(0).Cidade.value_counts()[:11]

def Plotar_Pie_Charts(dados):
    
    sn.set()
    corzinhas = ['b', 'g', 'r', 'c', 'y', 'm']
    
    frequencia = dados['Atividade_Emissor'].value_counts(dropna=True, sort=True)
    df = frequencia.rename_axis('unique_values').reset_index(name='counts')
    df2 = df[:5].copy()
    new_row = pd.DataFrame(data = {
        'unique_values' : ['OUTROS'],
        'counts' : [df['counts'][5:].sum()]
    })
    df2 = pd.concat([df2, new_row])
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plot = df2['counts'].plot(kind = 'pie', labels = df2['unique_values'],
              autopct='%1.0f%%', radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('')
    mpl.rcParams['font.size'] = 9.0
    plot.axis("equal")
    plt.tight_layout()
    plt.show()
    fig = plot.get_figure()
    fig.savefig('pie_atv.png', dpi = 250, bbox_inches='tight')
    
    plt.close('all')
    dados_EMAIL = dados[(dados.PossuiFaturaPorEmail == 1)]
    #dados_SMS = dados[(dados.PossuiSMSAlerta == 1)]
    
    frequencia = dados_EMAIL['Atividade_Emissor'].value_counts(dropna=True, sort=True)
    df = frequencia.rename_axis('unique_values').reset_index(name='counts')
    df2 = df[:5].copy()
    new_row = pd.DataFrame(data = {
        'unique_values' : ['OUTROS'],
        'counts' : [df['counts'][5:].sum()]
    })
    df2 = pd.concat([df2, new_row])
    
    
    plot = df2['counts'].plot(kind = 'pie', labels = df2['unique_values'],
              autopct='%1.0f%%', colors = corzinhas,
              radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('')
    plot.axis("equal")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.tight_layout()
    fig = plot.get_figure()
    plt.draw()
    fig.savefig('pie_atv_email.png', dpi = 250, bbox_inches='tight')
    plt.close('all')
    

    
    #dados_EMAIL = dados[(dados.PossuiFaturaPorEmail == 1)]
    dados_SMS = dados[(dados.PossuiSMSAlerta == 1)]
    
    frequencia = dados_SMS['Atividade_Emissor'].value_counts(dropna=True, sort=True)
    df = frequencia.rename_axis('unique_values').reset_index(name='counts')
    df2 = df[:5].copy()
    new_row = pd.DataFrame(data = {
        'unique_values' : ['OUTROS'],
        'counts' : [df['counts'][5:].sum()]
    })
    df2 = pd.concat([df2, new_row])
    
    
    plot = df2['counts'].plot(kind = 'pie', labels = df2['unique_values'],
              autopct='%1.0f%%', colors = corzinhas,
              radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('')
    plot.axis("equal")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.tight_layout()
    fig = plot.get_figure()
    plt.draw()
    fig.savefig('pie_atv_sms.png', dpi = 250, bbox_inches='tight')
    
    
    
    plt.close('all')
    dados_nenhum = dados[(dados.PossuiSMSAlerta == 0) & (dados.PossuiFaturaPorEmail == 0)]
    
    frequencia = dados_nenhum['Atividade_Emissor'].value_counts(dropna=True, sort=True)
    df = frequencia.rename_axis('unique_values').reset_index(name='counts')
    df2 = df[:5].copy()
    new_row = pd.DataFrame(data = {
        'unique_values' : ['OUTROS'],
        'counts' : [df['counts'][5:].sum()]
    })
    df2 = pd.concat([df2, new_row])
    
    
    plot = df2['counts'].plot(kind = 'pie', labels = df2['unique_values'],
              autopct='%1.0f%%', colors = corzinhas,
              radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('')
    plot.axis("equal")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.tight_layout()
    fig = plot.get_figure()
    plt.draw()
    fig.savefig('pie_nenhum.png', dpi = 250, bbox_inches='tight')
    

def serie_temporal():
    sn.set()
    serie = pd.read_csv('serie_temporal.csv')
    plot = serie.plot(style = 'o-')
    plt.xticks(np.arange(len(serie.index)), serie.Meses)
    plt.xlabel('Meses')
    plt.ylabel('Média de compras')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig = plot.get_figure()
    fig.tight_layout()
    plt.draw()
    fig.savefig('serie.png', dpi = 250, bbox_inches='tight')
   # plt.show()

def PlotarBoxplotsFatura(dados):
    df = dados[(dados['PossuiFaturaPorEmail'] == 1)]
    not_df = dados[(dados['PossuiFaturaPorEmail'] == 0)]
    
    X = df[df.columns.difference(['PossuiFaturaPorEmail',
                                  'PossuiSMSAlerta'])]
    
    not_X = not_df[not_df.columns.difference(['PossuiFaturaPorEmail',
                                              'PossuiSMSAlerta'])]
    sn.set()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plot1 = X.boxplot(column = ['Idade'], ax=axes[0])
    axes[0].set_title('Clientes com Fatura por e-mail')
    plot2 = not_X.boxplot(column = ['Idade'], ax=axes[1])
    axes[1].set_title('Clientes sem Fatura por e-mail')
    
    fig.savefig('boxplot_idade_Fatura.png', dpi = 250, bbox_inches='tight')

def PlotarBoxplotsSMS(dados):
    df = dados[(dados['PossuiSMSAlerta'] == 1)]
    not_df = dados[(dados['PossuiSMSAlerta'] == 0)]
    
    X = df[df.columns.difference(['PossuiFaturaPorEmail',
                                  'PossuiSMSAlerta'])]
    
    not_X = not_df[not_df.columns.difference(['PossuiFaturaPorEmail',
                                              'PossuiSMSAlerta'])]
    sn.set()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plot1 = X.boxplot(column = ['Idade'], ax=axes[0])
    axes[0].set_title('Clientes com SMS alerta')
    plot2 = not_X.boxplot(column = ['Idade'], ax=axes[1])
    axes[1].set_title('Clientes sem SMS alerta')
    
    fig.savefig('boxplot_idade_SMS.png', dpi = 250, bbox_inches='tight')
    

def selecionar_importantes(dados):
    
    #separar classes das características
    dadosY = dados[['PossuiFaturaPorEmail', 'PossuiSMSAlerta']]
    dadosX = dados[dados.columns.difference(['PossuiFaturaPorEmail', 
                                             'PossuiSMSAlerta'])]
    
    #inicializar modelo
    clf = RandomForestClassifier()
    
    #codificar dados categóricos
    for column in dadosX:
        if dadosX[column].dtype == type(int):
            le = preprocessing.LabelEncoder()
            dadosX[column] = le.fit_transform(dadosX[column])
    
    #ajustar modelo
    clf.fit(dadosX, dadosY)
    
    ##imprimir importancia de cada variável para o modelo ajustados
    #for feature in zip(dadosX.columns, clf.feature_importances_):
    #    print(feature)
    
    
    #selecionar variaveis com importância maior que 4.5%
    sfm = SelectFromModel(clf, threshold=0.045)
    sfm.fit(dadosX, dadosY)
    soma = []
    colunas = []
    for feature_list_index in sfm.get_support(indices=True):
        #print(dadosX.columns[feature_list_index])
        colunas.append(dadosX.columns[feature_list_index])
        soma.append(clf.feature_importances_[feature_list_index])
    #soma = np.array(soma)
    #np.sum(soma) #imprimir importância das variáveis
    #X_new = sfm.transform(dadosX)
    #X_new.reset_index(drop=True, inplace=True)
    #dadosY.reset_index(drop=True, inplace=True)
    
    X_new = dados[colunas]
    bigdata = pd.concat([X_new, dadosY], axis = 1)

    return bigdata

def Selecionar1000_Para_Email(dados):
    dados_ = pandas.get_dummies(dados,drop_first=True)
    
    df = dados_[(dados_['PossuiFaturaPorEmail'] == 1)]
    not_df = dados_[(dados_['PossuiFaturaPorEmail'] == 0)]

    #Y = df[['PossuiFaturaPorEmail']]
    X = df[df.columns.difference(['PossuiFaturaPorEmail',
                                  'PossuiSMSAlerta'])]
    
    not_X = not_df[not_df.columns.difference(['PossuiFaturaPorEmail',
                                              'PossuiSMSAlerta'])]
    
    centroide = np.array(np.mean(X))

    [m, n] = not_X.shape
    
    aux = not_X.values
    
    dist = []
    for i in range(0,m):
        dist.append(np.linalg.norm(centroide - aux[i]))
   
    indices = np.argsort(dist)[:1000]
    dados_.index.name = None
    
    Contas_selecionadas = []
    for i in range(0,1000):
        pd = dados_.iloc[[indices[i]]].index.tolist()[0]
        Contas_selecionadas.append(pd)
        
    Contas_selecionadas = np.array(Contas_selecionadas).astype(int)
    
    return Contas_selecionadas

    
def Selecionar1000_Para_SMS(dados):

    dados_ = pandas.get_dummies(dados,drop_first=True)

    df = dados_[(dados_['PossuiSMSAlerta'] == 1)]
    not_df = dados_[(dados_['PossuiSMSAlerta'] == 0)]
    
    #Y = df[['PossuiFaturaPorEmail']]
    X = df[df.columns.difference(['PossuiFaturaPorEmail',
                                  'PossuiSMSAlerta'])]
    
    not_X = not_df[not_df.columns.difference(['PossuiFaturaPorEmail',
                                              'PossuiSMSAlerta'])]
    
    centroide = np.array(np.mean(X))

    [m, n] = not_X.shape
    
    aux = not_X.values
    
    dist = []
    for i in range(0,m):
        dist.append(np.linalg.norm(centroide - aux[i]))
   
    indices = np.argsort(dist)[:1000]
    dados_.index.name = None
    
    Contas_selecionadas = []
    for i in range(0,1000):
        pd = dados_.iloc[[indices[i]]].index.tolist()[0]
        Contas_selecionadas.append(pd)
        
    Contas_selecionadas = np.array(Contas_selecionadas).astype(int)
    
    return Contas_selecionadas




def ModeloParaSugerirOuSMSOuEMAIL(dados):
    
    #Conjunto de Treinamento
    df = dados[(dados['PossuiSMSAlerta'] == 1) | (dados['PossuiFaturaPorEmail'] == 1)]
    #Conjunto de Testes
    not_df = dados[(dados['PossuiSMSAlerta'] == 0) & (dados['PossuiFaturaPorEmail'] == 0)]
    
    X = df[df.columns.difference(['PossuiFaturaPorEmail',
                                  'PossuiSMSAlerta'])]
    not_X = not_df[not_df.columns.difference(['PossuiFaturaPorEmail',
                                              'PossuiSMSAlerta'])]
    
    m, n = df.shape
    Y = np.zeros([m, 1])
    for i in range(0,m):
        if (df['PossuiSMSAlerta'].iloc[i]  == 1) & (df['PossuiFaturaPorEmail'].iloc[i]  == 1):
            Y[i] = 2
        elif (df['PossuiSMSAlerta'].iloc[i]  == 1) & (df['PossuiFaturaPorEmail'].iloc[i]  == 0):
            Y[i] = 1
        else:
            Y[i] = 2
            
    #Y.tofile("Y.csv",sep="\n")
    
    X = pd.get_dummies(X,drop_first=True)
    salvar_colunas = X.columns
    X = np.array(X)
    Y = np.array(Y)
    
    #fazer PCA
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    pca = PCA(n_components = 10)
    #Apenas 2% da variância, mas meu PC não suporta muito mais que isso devido a memória
    #pca.explained_variance_ratio_.cumsum()
    
    principalComponents = pca.fit_transform(X)
    
    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C),
              tree.DecisionTreeClassifier(),
              RandomForestClassifier(),
              MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
              LogisticRegression())
    
    
    scores = []
    for clf in models:
        scores.append(np.mean
                      (np.array
                       (cross_val_score(clf, principalComponents, Y.ravel(), cv=10))))
     
    
    melhor_modelo = np.argmax(scores)
    
    modelo_final = models[melhor_modelo]
    
    #modelo_final = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
      
    modelo_final.fit(principalComponents,Y.ravel())
    
    #Prever para o conjunto de treinamento
    Y_p = modelo_final.predict(principalComponents)
    matrix = confusion_matrix(Y, Y_p)
    print(matrix)
    report = classification_report(Y, Y_p)
    print(report)
    
    #Prever pro conjunto de testes
    Y_testes = prever(not_X, modelo_final, pca, sc, salvar_colunas)
    Y_testes = pd.DataFrame(Y_testes, columns = ['Classe'], index = not_X.index.values)
    Y_testes.index_name = 'IdConta'
    Y_testes.to_csv('predições_conj_treinamento.csv', index_label = 'IdConta')
    
    testes = not_X
    testes['Classe'] = Y_testes
    
    #corr = testes.corr()
    #sn_plot = sn.heatmap(round(corr,2), annot=True, cmap="coolwarm",fmt='.2f',
    #             linewidths=.05)
    #fig = sn_plot.get_figure()
    #fig.savefig('correlacao_Testes.png', dpi = 250, bbox_inches='tight',transparent=True)
    
    frequencia = Y_testes['Classe'].value_counts(dropna=True, sort=True)
    sn.set()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,)
    plot = frequencia.plot(kind = 'pie', autopct='%1.0f%%', radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('Classe prevista')
    mpl.rcParams['font.size'] = 9.0
    plot.axis("equal")
    plt.tight_layout()
    plt.show()
    fig = plot.get_figure()
    fig.savefig('pie_conjunto_teste.png', dpi = 250, bbox_inches='tight',transparent=True)
    plt.close('all')
    
    
    #Plotar correlação das previs
    #plot = sn.pairplot(df, hue='Classe', size=2.5, vars = colunas_numericas)
    #plot.savefig("paiplot_SMS.png",dpi = 250, bbox_inches='tight',transparent=True)
    
    

    

    return modelo_final, pca, sc, salvar_colunas



def prever(exemplo, modelo_final, pca, sc, salvar_colunas):
    
    exemplo = pd.get_dummies(exemplo,drop_first=False)
    
    add_missing_dummy_columns(exemplo, salvar_colunas)
    
    exemplo = exemplo[salvar_colunas]
    exemplo = sc.transform(exemplo)
    exemplo = pca.transform(exemplo)
    
    predicao = modelo_final.predict(exemplo)
    
    return predicao


def mais_plots(dados):
    df = dados[(dados['PossuiSMSAlerta'] == 1)]
    #df2 = dados[(dados['PossuiFaturaPorEmail'] == 1)]
    #not_df = dados[(dados['PossuiSMSAlerta'] == 0)]
    #not_df2 = dados[(dados['PossuiFaturaPorEmail'] == 0)]
    
    
    corr = dados.corr()
    sn_plot = sn.heatmap(round(corr,2), annot=True, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
    fig = sn_plot.get_figure()
    fig.savefig('correlacao.png', dpi = 250, bbox_inches='tight',transparent=True)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,)
    plot = df['PossuiEmailCadastrado'].value_counts().plot(kind = 'pie', autopct='%1.0f%%', radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('PossuiEmailCadastrado')
    plt.title('Clientes que possuem SMS alerta')
    mpl.rcParams['font.size'] = 9.0
    plot.axis("equal")
    plt.tight_layout()
    fig = plot.get_figure()
    fig.savefig('pie_email_SMS.png', dpi = 250, bbox_inches='tight',transparent=True)
    
    



def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0

def Pie_Gerais(dados):
    frequencia = dados['Sexo'].value_counts(dropna=True, sort=True)
    df = frequencia.rename_axis('unique_values').reset_index(name='counts')
    sn.set()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plot = df['counts'].plot(kind = 'pie', labels = df['unique_values'],
              autopct='%1.0f%%', radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('')
    mpl.rcParams['font.size'] = 9.0
    plot.axis("equal")
    plt.tight_layout()
    plt.show()
    fig = plot.get_figure()
    fig.savefig('pie_sexo.png', dpi = 250, bbox_inches='tight',transparent=True)
    
    
    
    #----------------------------------------------------------------------------------
    
    frequencia = dados['PossuiSMSAlerta'].value_counts(dropna=True, sort=True)
    
    df = frequencia.rename_axis('unique_values').reset_index(name='counts')
    sn.set()
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    
    plot = df['counts'].plot(kind = 'pie', labels = ['NaoPossuiSMSAlerta', 'PossuiSMSAlerta'],
              autopct='%1.0f%%', radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('')
    mpl.rcParams['font.size'] = 9.0
    plot.axis("equal")
    plt.tight_layout()
    plt.show()
    fig = plot.get_figure()
    fig.savefig('pie_SMSalerta_geral.png', dpi = 250, bbox_inches='tight',transparent=True)
    
    #----------------------------------------------------------------------------------
    frequencia = dados['PossuiFaturaPorEmail'].value_counts(dropna=True, sort=True)
    
    df = frequencia.rename_axis('unique_values').reset_index(name='counts')
    sn.set()
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    
    plot = df['counts'].plot(kind = 'pie', labels = ['NaoPossuiFaturaPorEmail', 'PossuiFaturaPorEmail'],
              autopct='%1.0f%%', radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('')
    mpl.rcParams['font.size'] = 9.0
    plot.axis("equal")
    plt.tight_layout()
    plt.show()
    fig = plot.get_figure()
    fig.savefig('pie_Email_geral.png', dpi = 250, bbox_inches='tight',transparent=True)
    
    
    #----------------------------------------------------------------------------------
    frequencia = dados['PossuiCartaoAdicional'].value_counts(dropna=True, sort=True)
    
    df = frequencia.rename_axis('unique_values').reset_index(name='counts')
    sn.set()
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plot = df['counts'].plot(kind = 'pie', labels = ['NaoPossuiCartaoAdicional', 'PossuiCartaoAdicional'],
              autopct='%1.0f%%', radius=1-0.4, wedgeprops=dict(width=0.4, edgecolor='w'))
    pylab.ylabel('')
    mpl.rcParams['font.size'] = 9.0
    plot.axis("equal")
    plt.tight_layout()
    plt.show()
    fig = plot.get_figure()
    fig.savefig('pie_PossuiCartaoAdicional_geral.png', dpi = 250, bbox_inches='tight',transparent=True)


def main():
    #ler e primeiros tratamentos dos dados
    dados = Ler_Tratar_Dados()

    #separar os conjuntos de caracteristas com mais de 85% de importância
    dados = selecionar_importantes(dados)
    
    #Plots e mais plots
    #dados.to_csv('novos_dados.csv')
    #Plotar_Pie_Charts(dados)
    #Plotar_Venn(dados)
    #Pie_Gerais(dados)
    #mais_plots(dados)
    #Plotar_Histogramas_TempoNaFB(dados)
    #Plotar_Histogramas(dados)
    #serie_temporal()
    
    #Selecionar 1000 clientes para ofertar cada produto
    SMS_1000 = Selecionar1000_Para_SMS(dados)
    SMS_1000.tofile("SMS_1000.csv",sep=",")
    
    
    
    EMAIL_1000 = Selecionar1000_Para_Email(dados)
    EMAIL_1000.tofile("EMAIL_1000.csv",sep=",")
    
    #Plotar Venn para os clientes selecionados
    sn.set()
    venn2([set(SMS_1000),set(EMAIL_1000)], set_labels = ('SMS_1000', 'EMAIL_1000'))
    plt.savefig('Venn_Sugeridos.png', dpi = 250)
    plt.close('all')
    #selecionar modelo e aplicar no conjunto de testes
    modelo_final, pca, sc, salvar_colunas = ModeloParaSugerirOuSMSOuEMAIL(dados)
    

    ##Exemplo de predição
    exemplo = pd.read_csv('exemplo.csv')
    predicao = prever(exemplo, modelo_final, pca, sc, salvar_colunas)
    print(predicao)
    
if __name__ == '__main__':
    main()
