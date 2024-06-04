import pandas as pd

# conversion of letter data set 

path = './data/letter/letter_ori.csv'

df = pd.read_csv(path, header=None)
print(df.head())

alphabet_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
print(alphabet_list)

df = df[[col for col in df.columns if col != 0] + [0]]
print(df.head())

path = './data/letter/letter.csv'
df.to_csv(path, header=False, index=False)

df = pd.read_csv(path, header=None)
print(df[16])
# count = 1
# for letter in alphabet_list:
#     print(letter, count)
    
#     count += 1

s = df[16]
for i, letter in enumerate(alphabet_list):
    print(letter, i)
    df[16] = df[16].str.replace(letter, str(i))
    
    
df[16] = df[16].astype(int)
print(df.head(10))
print(df.info())

path = './data/letter/letter.csv'
df.to_csv(path, header=False, index=False)

