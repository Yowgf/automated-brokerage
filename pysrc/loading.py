import pandas as pd
import numpy as np

dataframe = pd.read_csv("./data/home_data.csv")
initialCols = list(dataframe.columns)

def showInitialCols():
    print("{} colunas".format(len(initialCols)))
    print(initialCols)
