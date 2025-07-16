import pandas as pd

def df_add_row(df, d):
    d = d.copy() # just to be safe
    for x,y in d.items():
        d[x] = [y]
    d = pd.DataFrame(d)

    #if d.empty:
    #    return df
    #if df.empty:
    #    return d

    return pd.concat([df, d], ignore_index=True)
