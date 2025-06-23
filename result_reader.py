import pandas as pd

def read_PnL_results(_results: list | pd.Series):

    if isinstance(_results, list):
        print("Đổi kết quả sang pandas Series")
        results = pd.Series(_results)
    else:
        results = _results.copy()
    profits = results[results > 0].dropna()
    breakeven = results[results == 0].dropna()
    loss = results[results < 0].dropna()

    for name, i in [("Profits", profits), ("Break-even", breakeven), ("Loss", loss), ("Total results", results)]:
        if len(i) == 0:
            print(f"{name:<15}: count = 0 = 0.0%, mean = nan, max = nan, min = nan")
        else:
            print(f"{name:<15}: count = {len(i)} = {len(i)/len(results)*100:.1f}%, mean = {i.values.mean():.2f}, "
              f"max = {i.values.max():.2f}, min = {i.values.min():.2f}")
    
    return {
        "profit_count": len(profits),
        "breakevent_count": len(breakeven),
        "loss_count": len(loss),
        "total_count": len(results),
        "avg_PnL": results.values.mean()
    }

def main():
    results = pd.read_csv("total_results_v4.csv", index_col=0)
    read_PnL_results(results['PnL'])

if __name__ == "__main__":
    main()