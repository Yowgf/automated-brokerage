from .loading import *

# Auxilia a largar colunas de dataframe
def dropcol(col):
    if not col in dataframe.columns:
        raise AttributeError("Column not in dataframe")
        
    dataframe.drop(col, axis=1, inplace=True)

def dropUselessCols():
    # Nao precisamos do id
    dropcol("id")
    # Largando coluna de datas. Esta informacao nao ajuda em nada,
    #   pois nao buscamos construir uma serie temporal, e os valores
    #   sao todos muito recentes.
    dropcol("date")
    # Largando latitude, longitude, zipcode - nao precisamos
    #   de dados geograficos.
    dropcol("lat")
    dropcol("long")
    dropcol("zipcode")

def showFinalCols():
    datacols = list(dataframe.columns) # Manteremos estas
    print("{} colunas restantes".format(len(datacols)))
    print("Removidas ", [col for col in initialCols if col not in datacols])
    print("\n", datacols)
        
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

    assert(all(attvalues <= 2015))
    print("Ano de renovacao <= 2015")
    
def normalize(df):
    df -= df.min()
    df /= df.max()
