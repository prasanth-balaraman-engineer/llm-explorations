from dotenv import load_dotenv
from langchain.chains import load_chain

load_dotenv()

if __name__ == "__main__":
    llm_math_chain = load_chain("lc://chains/llm-math/chain.json")
    print(llm_math_chain.run("What is 42 raised to the power 0.4673?"))
