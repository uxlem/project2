import pandas as pd

results = pd.read_csv("total_results.csv", index_col=0)

profits = results[results > 0].dropna()
breakeven = results[results == 0].dropna()
loss = results[results < 0].dropna()

print(f"{len(profits)}, {len(breakeven)}, {len(loss)}")
