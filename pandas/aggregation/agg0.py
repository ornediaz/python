import pandas as pd

def main():
    df = pd.DataFrame({
        "First Name": ["John", "Jane", "Bob", "John"],
        "Last Name": ["Doe", "Smith", "Johnson", "Doe"],
        "Time": [1, 2, 3, 4]
    })

    df["Full Name"] = df["First Name"] + " " + df["Last Name"]

    result = (
        df
        .groupby("Full Name", as_index=False)
        .agg(
            Count=("Time", "count"),
            Mean_Time=("Time", "mean"),
            **{
                "Median Time": ("Time", "median"),
                "Average Time": ("Time", "mean")
            }
        )
    )

    print(result)

main()
