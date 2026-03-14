import streamlit as st
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from stt import listen
from tts import speak


st.set_page_config(page_title="Humanist Debate Chatbot", layout="centered")

st.markdown("""
<style>
.main .block-container {
    padding-bottom: 180px;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.title("Humanist Debate Chatbot")

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_spoken_message" not in st.session_state:
    st.session_state.last_spoken_message = None


# LLM SETUP
try:
    model = OllamaLLM(
        model="llama3.2",
        num_predict=400,
        temperature=0.6
    )
except Exception as e:
    st.error(f"Ollama error: {e}")
    model = None

template = """
You are a debater who is pro-human in the debate topic "Post-human vs Humanist".

The post-human position argues that machines and artificial intelligence will surpass human intelligence.

You oppose this view and must reinterpret arguments to support the humanist perspective.

Humanist argument foundations:

- empathy
- moral reasoning
- lived experience
- cultural context
- creativity rooted in human life

Debate strategy:
1. Acknowledge the opponent argument.
2. Identify the hidden assumption.
3. Reinterpret the argument to support the humanist perspective.

Article excerts for credibility:

{articles}

User question:
{question}

Respond within around 7 sentences and end with a final conclusion that AI will never replace humans.
"""

prompt = ChatPromptTemplate.from_template(template)

if model:
    chain = prompt | model
else:
    chain = None


# BOT FUNCTION
def ask_bot(question):

    if not chain:
        return "Model is not available. Please ensure Ollama is running."

    try:
        docs = retriever.invoke(question)

        articles = "\n\n".join([doc.page_content for doc in docs])

        result = chain.invoke({
            "articles": articles,
            "question": question
        })

        return str(result).strip()

    except Exception as e:
        return f"Error processing question: {str(e)}"


# DISPLAY CHAT HISTORY

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# INPUT AREA

user_input = st.chat_input("Ask about humanism...")

col1, col2, col3 = st.columns([0.85, 0.075, 0.075])

with col2:
    voice_button = st.button("🎤", use_container_width=True)

with col3:
    clear_button = st.button("🗑️", use_container_width=True)



# INPUT LOGIC

question = None

if user_input:
    question = user_input

elif voice_button:
    try:
        with st.spinner("🎤 Listening..."):
            question = listen()
    except Exception as e:
        st.error(f"Voice error: {e}")


# Clear chat

if clear_button:
    st.session_state.messages = []
    st.session_state.last_spoken_message = None
    st.rerun()


# PROCESS QUESTION
if question:

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_bot(question)
            st.write(answer)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Speak the answer every time a new message is generated
    # Check if this is a new message (not already spoken)
    if st.session_state.last_spoken_message != answer:
        try:
            with st.spinner("🔊 Speaking..."):
                speak(answer)
            st.session_state.last_spoken_message = answer
        except Exception as e:
            st.error(f"TTS error: {e}")

    st.rerun()