from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryMemory

load_dotenv()


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent total of {cb.total_tokens} tokens.')
    return result


questions = [
    'Good morning AI!',
    'My interest here is to explore the potential of integrating Large Language Models with external knowledge',
    'I just want to analyze the different possibilities. What can you think of?',
    'Which data source types could be used to give context to the model?',
    'What is my aim again?',
]

if __name__ == '__main__':
    llm = OpenAI(model_name='text-davinci-003', temperature=0)
    chain = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))

    for question in questions:
        count_tokens(chain, question)

    print(chain.memory.buffer)
