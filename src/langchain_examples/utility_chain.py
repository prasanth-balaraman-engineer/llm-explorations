from dotenv import load_dotenv
from langchain.chains import LLMMathChain
from langchain.llms import OpenAI

from utils.openapi import count_tokens

load_dotenv()

if __name__ == "__main__":
    llm = OpenAI(model_name="text-davinci-003")
    llm_chain = LLMMathChain.from_llm(llm, verbose=True)
    result = count_tokens(
        chain=llm_chain, query="What is 13 raised to the .3432 power?"
    )
    print(result)
