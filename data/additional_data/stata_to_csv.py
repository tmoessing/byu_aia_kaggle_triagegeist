import pandas as pd
data = pd.read_stata('data/ed2022-stata/ed2022-stata.dta', convert_categoricals=False).to_csv('data/ed2022.csv')
