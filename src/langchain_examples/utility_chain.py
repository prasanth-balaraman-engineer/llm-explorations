from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMMathChain
from langchain.llms import OpenAI

load_dotenv()


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result


if __name__ == '__main__':
    llm = OpenAI(model_name='text-davinci-003')
    llm_chain = LLMMathChain.from_llm(llm, verbose=True)
    result = count_tokens(chain=llm_chain, query='What is 13 raised to the .3432 power?')
    print(result)
