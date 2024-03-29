import pandas as pd #importa a biblioteca pandas endereço (pandas.pydata.org)
import matplotlib.pyplot as plt #importa a biblioteca de gráficos pyplot (https://matplotlib.org/)
import numpy as np #importa a biblioteca matemática Numpy (numpy.org)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #importa as bibliotecas de métricas para avalição da qualidade dos modelos
from sklearn.linear_model import LogisticRegression #importa a biblioteca para análise linear da biblioteca sklearn
from sklearn.metrics import confusion_matrix # importa a biblioteca de matriz de confusão
from sklearn.metrics import recall_score # importa a biblioteca para sensibilidade e especifidade
from sklearn.metrics import precision_recall_fscore_support #importa a a biblioteca para avaliar a precisão do modelo 
from sklearn.model_selection import train_test_split #importa a biblioteca para treino do modelo e teste do Modelo 
from sklearn.model_selection import KFold #importa a bliblioteca para fazer a validação cruzada dos conjuntos de treino


df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv') #read_csv converte o formato csv do arquivo de dados no formato Pandas Dataframe. nrows delimita o número de linhas a ler, útil para grandes conjuntos de dados
print(df.head()) #imprime no console as colunas do dataframe
#print(df.describe()) #Retorna uma tabela de estatísticas sobre as colunas usado para entender os conjuntos de dados Count: Este é o numero de linhas que possuem algum valor. Neste caso, todos os passageiros possuem um valor em cada coluna, então o valor é 887 (o total de número de passageiros). Mean: Recupera a média padrão. Std: é uma abreviação para desvio padrão. É uma medida da dispersão dos dados. Min: O menor valor.25%: O vigésimo quinto porcentil. 50%: O quiquagésimo porcentil, também chamado de mediana. 75%: O septuagésimo quinto porcentil. Max: O maior valor.
#col =df['Fare'] #Seleciona uma unica coluna para ser impressa no console
#print(col)#imprime a coluna no console constitui uma Panda Series
#small_df = df[['Age','Sex','Survived']] #Faz uma seleção de dados dentro do DataFrame e une como um conjunto de dados menor
# print(small_df.head()) #Imprime o conjunto de dados menor
# df['male'] = df['Sex'] == 'male' #Cria uma nova coluna no DataFrame com o nome declarado, faz uma comparação lógica e retorna o valor da comparação para a nova coluna. Numpy é um pacote do Python para a manipulação de listas e tabelas numéricas. Podemos usar ela para realizar muitos cálculos estatísticos. Nós chamamos de lista ou tabela de dados uma matriz numpy. Geralmente iremos tomar um dado de um pandas DataFrame e colocar em uma matriz do tipo numpy. Pandas DataFrames são ótimos porque possuem os nomes das colunas e outros textos que fazem delas legíveis aos humanos. Apesar disso não são a forma ideal para realização de cálculos. As matrizes numpy são menos fáceis de ler para humanos, porém possuem o formato necessário para a computação dos dados.
# print("Agora vamos converter uma das colunas do DataFrame em uma matriz Numpy: ")
# print(df['Fare'].values) #Usa a biblioteca NumPy para converter a coluna Fare em um variável do tipo vetor (array)
# print("Também podemos converter mais colunas ao mesmo tempo de Dataframe para Numpy: ")
# print(df[['Pclass','Fare','Age']].values)#usa a mesma idéia da linha acima mas produz uma matriz tridimensional (array)
# print("Além disso podemos gravar essa nova matriz em um arranjo de dados para ser manipulado: ")
# arr = df[['Pclass','Fare','Age']].values #usa também a mesma idéia mas grava a matriz gerada na variável arr
# print(arr.shape) #Imprime no console o número de linhas e colunas da matriz na variável arr
# #Selecionar dados específicos usando NumPy
# print(arr[0,1]) #imprime o dados na primeira linha e sequnda coluna do conjunto de dados arr
# print(arr[0]) # imprime a primeira linha do conjunto de dados arr
# print(arr[:,2])# imprime a terceira coluna de dados do conjunto arr
# #criar mascaras de teste
# mask = arr[:,2] < 18 # retorna apenas os dados que são menore que 18
# print(arr[mask])
# print("fim menor que 18")
# print(arr[arr[:,2] > 18]) # também pode ser escrito em uma linha, agora retornará o que for maior que 18
# print("fim maior que 18")
# print("Soma das idades menores que 18: ", arr[arr[:,2] < 18].sum())
# plt.scatter(df['Age'],df['Fare']) #plota um gráfico de dispersão com os dados de idade versus tarifa
# plt.xlabel('Age') # coloca uma legenda no eixo x
# plt.ylabel('Fare') # coloca uma legenda no eixo y
# plt.scatter(df['Age'],df['Fare'],c=df['Pclass']) #plota o mesmo gráfico anterior mas marcando com cores diferentes as diferentes classes
# plt.plot([0,80],[85,5]) #desenha uma linha no gráfico entre os dois pontos descritos
# plt.show(block=False)
# print("precione qualquer tecla para continuar")
# input()
# plt.close()
# X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# #X = df[['Fare','Age']].values
# y = df['Survived'].values
# print("imprimindo valores de x: ")
# print(X)
# print("imprimindo valores de y: ")
# print(y)

# #plt.scatter(df['Age'],df['Fare'],c=df['Pclass'])

# model = LogisticRegression()
# model.fit(X, y)
# print("imprimindo os coeficientes da reta e seu intercepto: ")
# print(model.coef_,model.intercept_)
# print("imprimindo a previsão segundo os dados de um passageiro: ") #Pclass 3, male, 22 years old, 1 sibling/spouse aboard, 0 parents/child aboard, paid $7.25
# print(model.predict([[3,True,22.0,1,0,7.25]]))
# print("[0] significa que o passageiro não sobreviveu, [1] que sobreviveu. ")
# print("fazendo a previsão para as 100 primeiras linhas de dados: ")
# print(model.predict(X[:100]))
# print("compare com os dados reais: ")
# print(y[:100])
# print("Nem todas as previsões estão corretas, mas a quantidade de acertos é alta")

# #Avaliação da qualidade do modelo
# y_pred = model.predict(X)
# print((y == y_pred).sum()) 
# print((y == y_pred).sum() / y.shape[0])
# print("O modelo apresenta: ", model.score(X, y)*100,"% de acertos!") 
# #Métricas do Modelo
# print("Essa é a exatidão do modelo: ",accuracy_score(y, y_pred)*100,"%")
# print("Essa é a precisão do modelo: ", precision_score(y,y_pred)*100,"%")
# print("Esse é o desvio do modelo: ", recall_score(y,y_pred)*100,"%")
# print("Essa a nota F1 do modelo: ", f1_score(y,y_pred)*100,"%")
# #imprimir a matriz de confusão do ModuleNotFoundError
# print("Essa é a matriz de confusão do modelo: ", confusion_matrix(y,y_pred))
# # primeira linha Actual negative, segunda linha Actual positive, primeira coluna predicted negative, segunda coluna predicted positive, 
# # Fazendo o treinamento do modelo, criando o train test para X e y
# X_train,X_test,y_train,y_test =train_test_split(X,y, random_state = 27) 
# # Vendo a forma dos atributos para descobrir seus tamanhos
# print("todos os dados: ", X.shape, y.shape)
# print("conjunto de treino: ", X_train.shape, y_train.shape)
# print("conjunto de teste: ", X_test.shape, y_test.shape)
# model = LogisticRegression() #reiniciando a regressão linear
# model.fit(X_train,y_train)
# print("Avaliação do modelo de teste: ", model.score(X_test,y_test)*100,"%")
# y_pred = model.predict(X_test)
# print("A exatidão do modelo é:", accuracy_score(y_test,y_pred)*100,"%")
# print("A precisão do modelo é:", precision_score(y_test,y_pred)*100,"%")
# print("Desvio do modelo: ", recall_score(y_test,y_pred)*100,"%")
# print("Nota de qualidade do modelo f1:", f1_score(y_test,y_pred)*100,"%")
# #X_train,X_test,y_train,y_test =train_test_split(X,y,random_state = 27) #Aplicando um fator para que o conjunto de dados gerados seja sempre homogêneo
# #print('X_train',X_train)
# #print('X_test',X_test)
# #Trabalhando com sensibilidade e especificidade
# sensitivity_score = recall_score
# print("Nota de sensibilidade do modelo: ", sensitivity_score(y_test, y_pred)*100,"%")
# #print("A precisão do modelo é de:", precision_recall_fscore_support(y_test,y_pred))
# def specificity_score(y_true, y_pred):
#     p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
#     return r[0]
# print("A nota de especificidade do modelo é: ",specificity_score(y_test, y_pred)*100,"%") 
# #Trabalhando com metas através de curvas de desvio
# model.fit(X_train,y_train)
# print("probabilidade predita: [MORTE, VIDA]")
# print(model.predict_proba(X_test))
# y_pred = model.predict_proba(X_test)[:,1] > 0.75
# print("Precisão:", precision_score(y_test,y_pred)*100, "%")
# print("Desvio: ", recall_score(y_test,y_pred)*100, "%")
# #Trabalhando com multiplos subconjuntos de treino e teste. O mesmo conjunto de dados é sempre subdividido em pontos diferentes. Primeiro vou reiniciar as variáveis

# X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# y = df['Survived'].values


# # construindo o modelo / building model
# model = LogisticRegression()
# model.fit(X_train, y_train)
# print("Treinando com diversos conjutos de treino: ")
# y_pred = model.predict(X_test)
# print("Exatidão: {0:.5f}".format(accuracy_score(y_test,y_test)))
# print("Precisão: {0:.5f}".format(precision_score(y_test,y_pred)))
# print("Desvio: {0:.5f}".format(recall_score(y_test,y_test)))
# print("Nota de qualidade do modelo f1: {0:.5f}".format(f1_score(y_test,y_pred)))
# #fazendo o mesmo com validação cruzada entre os conjuntos de treino para as seis primeiras linhas
# X_train,X_test,y_train,y_test =train_test_split(X,y, random_state = 27) 
# X = df[['Age', 'Fare']].values[:50]
# y = df['Survived'].values[:50]
# kf = KFold(n_splits=3, shuffle=True)
# splits = list(kf.split(X))
# first_split = splits[0]
# train_indices,test_indices = first_split
# print("training set indices:", train_indices)
# print("test set indices:", test_indices)
# print("Precione uma tecla para continuar...")
# input()
# X_train = X[train_indices]
# X_test = X[test_indices]
# y_train = y[train_indices]
# y_test = y[test_indices]
# print("X_train")
# print(X_train)
# print("y_train", y_train)
# print("X_test")
# print(X_test)
# print("y_test", y_test)
# print("Precione uma tecla para continuar...")
# input()
# print("o modelo agora será treinado com cinco subconjuntos: ")
# scores = []
# kf = KFold(n_splits=5, shuffle=True)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     model = LogisticRegression()
#     model.fit(X_train, y_train)
#     scores.append(model.score(X_test, y_test))
# print("Nota média do modelo:", np.mean(scores))
# final_model = LogisticRegression()
# final_model.fit(X,y)
