from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "superstore.csv"


def load_orders(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    try:
        orders = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        orders = pd.read_csv(path, encoding="cp1252")
    required = {"Order Date", "Sales"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    orders = orders.drop_duplicates().copy()
    orders["Order Date"] = pd.to_datetime(orders["Order Date"], errors="coerce")
    orders["Sales"] = pd.to_numeric(orders["Sales"], errors="coerce")
    orders = orders.dropna(subset=["Order Date", "Sales"])
    if "Category" in orders:
        orders["Category"] = orders["Category"].fillna("Unknown").astype(str)
    else:
        orders["Category"] = "All"
    return orders


def weekly_sales(
    category: str = "All",
    path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    orders = load_orders(path)
    if category != "All":
        orders = orders.loc[orders["Category"] == category]
    if orders.empty:
        raise ValueError(f"No sales rows found for category {category!r}")

    weekly = (
        orders.set_index("Order Date")["Sales"]
        .sort_index()
        .resample("W-SUN")
        .sum()
        .asfreq("W-SUN", fill_value=0)
    )
    return weekly.rename_axis("ds").rename("y").reset_index()


def available_categories(path: str | Path = DEFAULT_DATA_PATH) -> list[str]:
    orders = load_orders(path)
    return ["All", *sorted(orders["Category"].unique())]