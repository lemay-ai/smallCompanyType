import glob
import pandas as pd
from IPython.display import display
import smallCompanyType as s
b=s.SmallCompanyType()

files = glob.glob('TargetBusinesses_*.csv')

def getCategory(row):
    row["Type"]=b.getCompanyType(b.title_except(row["name"]))
    row["Subtype"]=b.getCompanySubtype(b.title_except(row["name"]))
    return row

for file in files:
    df = pd.read_csv(file)
    df=df.apply(getCategory, axis=1)
    print(df[['name','Type','Subtype']])
