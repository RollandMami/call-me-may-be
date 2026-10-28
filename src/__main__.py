from llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()

prompt = "hello"

with open(model.get_path_to_vocab_file(),"r", encoding="utf-8") as f:
    content = json.load(f)
    print(content["hello"])

encoded = model.encode(prompt)[0].tolist()

print(f"===================={encoded}")
