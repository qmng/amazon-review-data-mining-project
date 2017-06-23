import pandas as pd

newLine = [1,2,3]
df = pd.DataFrame([newLine])
df.to_csv(open('pandas_demo.csv', 'a'), index=False, header=False)
