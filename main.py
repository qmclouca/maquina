import pandas as pd #import pandas library
import matplotlib.pyplot as plt #importa a biblioteca de gráficos do python
import numpy as np

from sklearn.linear_model import LogisticRegression #importa a biblioteca para análise linear da biblioteca sklearn

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv') #read_csv converte o formato csv do arquivo de dados no formato Pandas Dataframe
print(df.head()) #imprime no console as colunas do dataframe
print(df.describe()) #Retorna uma tabela de estatísticas sobre as colunas usado para entender os conjuntos de dados
                     #Count: This is the number of rows that have a value. In our case, every passenger has a value for each of the columns, so the value is 887 (the total number of passengers).
                     #Mean: Recall that the mean is the standard average.
                     #Std: This is short for standard deviation. This is a measure of how dispersed the data is.
                     #Min: The smallest value
                     #25%: The 25th percentile
                     #50%: The 50th percentile, also known as the median.
                     #75%: The 75th percentile
                     #Max: The largest value
col =df['Fare'] #Seleciona uma unica coluna para ser impressa no console
print(col)#imprime a coluna no console constitui uma Panda Series
small_df = df[['Age','Sex','Survived']] #Faz uma seleção de dados dentro do DataFrame e une como um conjunto de dados menor
print(small_df.head()) #Imprime o conjunto de dados menor
df['male'] = df['Sex'] == 'male' #Cria uma nova coluna no DataFrame com o nome delarado faz uma comparação lógica e retorna o valor da comparação para a nova coluna
#Numpy is a Python package for manipulating lists and tables of numerical data. We can use it to do a lot of statistical calculations. We call the list or table of data a numpy array. We often will take the data from our pandas DataFrame and put it in numpy arrays. Pandas DataFrames are great because we have the column names and other text data that makes it human readable. A DataFrame, while easy for a human to read, is not the ideal format for doing calculations. The numpy arrays are generally less human readable, but are in a format that enables the necessary computation.
print(df['Fare'].values) #Usa a biblioteca NumPy para converter a coluna Fare em um variável do tipo vetor (array)
print(df[['Pclass','Fare','Age']].values)#usa a mesma idéia da linha acima mas produz uma matriz tridimensional (array)
arr = df[['Pclass','Fare','Age']].values #usa também a mesma idéia mas grava a matriz gerada na variável arr
print(arr.shape) #Imprime no console o número de linhas e colunas da matriz na variável arr
#Selecionar dados específicos usando NumPy
print(arr[0,1]) #imprime o dados na primeira linha e sequnda coluna do conjunto de dados arr
print(arr[0]) # imprime a primeira linha do conjunto de dados arr
print(arr[:,2])# imprime a terceira coluna de dados do conjunto arr
#criar mascaras de teste
mask = arr[:,2] < 18 # retorna apenas os dados que são menore que 18
print(arr[mask])
print("fim menor que 18")
print(arr[arr[:,2] > 18]) # também pode ser escrito em uma linha, agora retornará o que for maior que 18
print("fim maior que 18")
print("Soma das idades menores que 18: ", arr[arr[:,2] < 18].sum())
plt.scatter(df['Age'],df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')

#X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
#y = df['Survived'].values


#plt.scatter(df['Age'],df['Fare'],c=df['Pclass'])

#model = LogisticRegression()
#model.fit(X, y)
#
#y_pred = model.predict(X)
#print((y == y_pred).sum())
#print((y == y_pred).sum() / y.shape[0])
#print(model.score(X, y))