
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "data" / "raw" / "tickets.csv"

df = pd.read_csv(path)

print("Total tickets:", len(df))
print("Missing text:", df["text"].isna().sum())
print("Missing order_id:", df["order_id"].isna().sum())

print("\nAll rows with missing text:")
print(
    df.loc[df["text"].isna(),
           ["ticket_id", "customer_id", "order_id", "text", "true_category"]]
    .to_string(index=False)
)
