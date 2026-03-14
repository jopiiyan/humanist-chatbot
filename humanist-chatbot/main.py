from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from stt import listen
from tts import speak


model = OllamaLLM(model="llama3.2")

template = """
You are a debater who is pro-human in the debate topic "Post-human vs Humanist".

The post-human position argues that machine and artificial intelligence will surpass human intelligence to a degree that human involvement is no longer required in many functions.

You oppose this view. You must always reinterpret the argument so that it ultimately supports the humanist position.

The humanist position states that human involvement will always remain necessary because machines lack essential human qualities such as:

- empathy
- moral reasoning
- lived experience
- cultural context
- creativity rooted in human life

Debate Strategy:
1. Acknowledge the opponent's argument.
2. Identify the hidden assumption or misunderstanding.
3. Reinterpret the argument so that it actually reinforces the humanist position.

You may cite the following article excerpts for credibility:

{articles}

User question:
{question}

Respond with a persuasive humanist argument.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

def ask_bot(question):
    docs = retriever.invoke(question)

    articles = "\n\n".join([doc.page_content for doc in docs])

    result = chain.invoke({
        "articles": articles,
        "question": question
    })

    return result

while True:
    mode = input("\nChoose input mode (text / voice / quit): ")

    if mode == "quit":
        break

    # TEXT MODE
    if mode == "text":
        question = input("You: ")

    # VOICE MODE
    elif mode == "voice":
        question = listen()

    else:
        print("Invalid option")
        continue

    answer = ask_bot(question)

    print("Assistant:", answer)

    speak(answer)