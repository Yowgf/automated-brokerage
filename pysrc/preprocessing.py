from .loading import *

# Auxilia a largar colunas de dataframe
def dropcol(col):
    if col in dataframe.columns:
        dataframe.drop(col, axis=1, inplace=True)

def showFinalCols():
    datacols = list(dataframe.columns) # Manteremos estas
    print("{} colunas restantes".format(len(datacols)))
    print("Removidas ", [col for col in initialCols if col not in datacols])
    print(datacols)
        
def checkDataRanges():
    # Sem entradas faltantes
    assert(dataframe.isnull().sum().sum() == 0)
    print("Zero valores faltantes")
    # Todos os valores maiores ou iguais a zero
    assert(all(dataframe.values.flatten() >= 0))
    print("Todos os valores sao nao-negativos")
    # Todos os anos de construcao estao entre 1900 e 2015
    att = "yr_built"
    assert(all(1900 <= dataframe[att]) and all(dataframe[att] <= 2015))
    print("Anos de contrucao entre 1900 e 2015")
    # Todos os anos de renovacao, se existem, sao maiores que os
    # anos de construcao
    att = "yr_renovated"
    attvalues = dataframe[att].values
    assert(all(np.logical_or(attvalues == 0, attvalues >= dataframe["yr_built"].values)))
    print("Ano de renovacao >= ano de contrucao")
    
