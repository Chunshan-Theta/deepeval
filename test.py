import pandas as pd
import re

# Load the CSV file into a DataFrame
df = pd.read_csv('goldenset.csv')

# Display the first few rows of the DataFrame
# print(df[['question','human_answer']].head())
print(df)

with open("./inputs.txt", "w") as f:
    for _, row in df[:200].iterrows():  # We don't need to use 'row[1]' directly
        # Extract 'question' and 'human_answer' for each row
        question = row['question']
        human_answer = row['human_answer']
        
        # Write formatted content to the file
        f.write(f"- text: '{question.replace("\n","\\n").replace("\'","\"")}'\n  true_answer: '{human_answer.replace("\n","\\n").replace("\'","\"")}'\n\n")