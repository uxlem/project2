import pandas as pd

results = pd.read_csv("total_results_s2.csv", index_col=0)

profits = results[results > 0].dropna()
breakeven = results[results == 0].dropna()
loss = results[results < 0].dropna()

for i in profits, breakeven, loss, results:
    print(f"count = {len(i)} = {len(i)/len(results)*100:.1f}%, mean = {i.values.mean():.2f}, "
          f"max = {i.values.max():.2f}, min = {i.values.min()}:.2f")

print(f"{profits.mean()}, {loss.mean()}")

print()