from datasets import load_dataset
import pandas as pd

ds = load_dataset("nbertagnolli/counsel-chat")
df = pd.DataFrame(ds["train"])
df.to_csv("counsel_chat.csv", index=False)
print("saved")