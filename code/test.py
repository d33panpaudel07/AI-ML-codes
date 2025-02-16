import numpy as np
import pandas as pd

df = {
    "Weight": [45, 55, 50, 60],
    "Name": ["Sam", "Jack", "John", "Tony"],
    "Age": [20, 30, 20, 22],
}

df = pd.DataFrame(df)
print(df)
