"""This file contains utilities that are not directly tied to the
data mining notebook flow.

"""

from .loading import dataframe

def generateDataDescription(outFileName):
    ddFile = open(outFileName, "w")

    for col in dataframe.columns:
        ddFile.write(f"{col}:\n")
        ddFile.write(f"  type: \"{dataframe[col].dtype.name}\"\n")
        ddFile.write(f"  range: [{dataframe[col].min()}, {dataframe[col].max()}]\n")
        ddFile.write(f"  description: \"none\"\n")

    ddFile.close()
