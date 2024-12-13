import pandas as pd

df = pd.read_csv('1129_錯題.csv')
df = df.dropna(subset=['学科'])
print(df)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("-"*10)

# liberal_arts = df[df['学科'].str.contains('文科')]
# science = df[df['学科'].str.contains('理科')]
# print(liberal_arts)
# print(science)
# print("-"*10)

# liberal_arts_error = liberal_arts[liberal_arts['llm答案是否正确（仅对「是否有效题目」中为“是”的作标注）'].str.contains('错误')]
# science_error = science[science['llm答案是否正确（仅对「是否有效题目」中为“是”的作标注）'].str.contains('错误')]
# print(liberal_arts_error)
# print(science_error)
# print("-"*10)
df_error = df[df['llm答案是否正确（仅对「是否有效题目」中为“是”的作标注）'].str.contains('错误')]
print(df_error)
print("-"*10)


# science_error.to_csv('1129_error_case.csv', index=False)
df_error.to_csv('error_case.csv', index=False)

