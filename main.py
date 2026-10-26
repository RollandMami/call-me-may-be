from llm_sdk import Small_LLM_Model

def main() -> None:
    
    model = Small_LLM_Model()
    #encoded_txt = model.encode("Hello, world!")
    #print(f"Encoded: {encoded_txt}")
    encoded_txt = [1, 2, 3, 4, 5]  # Example token IDs for demonstration
    print(f"Decoded: {model.decode(encoded_txt)}")

if __name__ == "__main__":
    main()
