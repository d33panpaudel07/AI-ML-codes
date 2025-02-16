import pandas as pd

df = pd.DataFrame(
    {
        "Weight": [45, 55, 50, 60],
        "Name": ["Sam", "Jack", "John", "Tony"],
        "Age": [20, 30, 20, 22],
    }
)

# create index
index_ = ["Row_1", "Row_2", "Row_3", "Row_4"]

df.index = index_
# print(df.iloc[0, 1])

print("Original Dataframe: ")
print(df)

# basically from column "Name" on row 4 select the value
# result = df.loc["Row_4", "Name"]
# print("\nSelected value at Row_4, Column 'Name': ")
# print(result)

# selecting all rows form specific columns
result1 = df.loc[:, ["Name", "Age"]]
print(result1)


# print(df)
