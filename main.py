import streamlit as st
import random
import requests
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
anthropic_api_key = os.getenv("CLAUDE_API_KEY")

if not anthropic_api_key:
    raise ValueError(
        "❌ Anthropic API key not found. Please set ANTHROPIC_API_KEY in your .env file."
    )
# ------------------ Logging Setup ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="kids_vocab_app.log",
    filemode="a",
)


# ------------------ Pydantic Schema ------------------
class WordInfo(BaseModel):
    meaning: str = Field(..., description="Simple meaning of the word in English")
    opposites: list[str] = Field(..., description="Three opposites of the word")
    similar_words: list[str] = Field(..., description="Three similar words")
    sentences: list[str] = Field(
        ..., description="Three short and simple sentences using the word"
    )
    image_prompt: str = Field(
        ..., description="A short description for an image illustration"
    )


# ------------------ LangChain + Claude ------------------
def get_word_details(word: str) -> WordInfo:
    """Generate structured word info using Claude."""
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929", temperature=0.6, api_key=anthropic_api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """
    You are a kind English tutor for kids aged 6–10.

    For the word "{word}", provide:
    1. A short and simple meaning.
    2. Three opposite words (antonyms).
    3. Three similar words (synonyms).
    4. Three very simple example sentences with 5 to 6 words. This is for Grade 1 to 3 kids
    5. A short, vivid description for an illustration image (e.g., "A happy sun smiling in the sky").

    Respond ONLY in JSON format:
    {{
        "meaning": "...",
        "opposites": ["...", "...", "..."],
        "similar_words": ["...", "...", "..."],
        "sentences": ["...", "...", "..."],
        "image_prompt": "..."
    }}
    """
    )

    message = prompt.invoke({"word": word})
    response = llm.invoke(message)
    response_text = response.content  # ✅ Use .content for final text
    logging.info(f"Raw response for '{word}': {response_text}")
    import re

    response_text = re.sub(
        r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.DOTALL
    )

    logging.info(f"Cleaned response for '{word}': {response_text}")

    try:
        data = WordInfo.parse_raw(response_text)
    except ValidationError as e:
        logging.error(f"Validation failed for '{word}': {e}")
        raise ValueError("AI response format error.")
    return data


# ------------------ Image via Unsplash ------------------
def get_image_for_word(description: str):
    """Fetch a kid-friendly image from Unsplash using the description."""
    try:
        # Unsplash Source API for quick random photos
        query = description.replace(" ", "+")
        image_url = f"https://source.unsplash.com/512x512/?{query},cartoon,colorful"
        # Quick check to ensure the URL is valid
        res = requests.get(image_url)
        if res.status_code == 200:
            return image_url
        else:
            return None
    except Exception as e:
        logging.error(f"Image fetch failed: {e}")
        return None


# ------------------ Streamlit UI ------------------
st.set_page_config(
    page_title="🪄 WonderWords",
    page_icon="🎓",
    layout="centered",
)
# ------------------ Header ------------------
st.markdown(
    """
    <div class="title-container">
        <h1></h1>
        <p></p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "Learn words with **meanings**, **opposites**, **similar words** and **sentences**!"
)
st.image("assets/header_bg.jpg", use_container_width=True)
# ------------------ Session State ------------------
if "history" not in st.session_state:
    st.session_state.history = {}
# ------------------ Input Section ------------------
st.markdown("***Enter a word to find its meaning, opposites, and examples.*** ")
word = st.text_input("", value="", max_chars=30, label_visibility="collapsed")

# ------------------ Explore Word ------------------
if st.button("✨ Explore Word", type="primary"):
    if not word.strip():
        st.warning("Please enter a word first!")
    else:
        word = word.strip().lower()
        with st.spinner("Thinking... 🧠"):
            try:
                info = get_word_details(word)
                image_url = get_image_for_word(info.image_prompt)
                st.session_state.history[word] = info.dict()

                # Display result
                st.success(f"Here's what I found about **{word}**!")

                tab1, tab2, tab3, tab4 = st.tabs(
                    [
                        "📖 Meaning",
                        "🚫 Opposites",
                        "🤝 Similar Words",
                        "🗣️ Sentences",
                    ]
                )

                with tab1:
                    st.write(f"**Meaning:** {info.meaning}")

                with tab2:
                    st.markdown("**Opposite Words:**")
                    for i, opp in enumerate(info.opposites, 1):
                        st.write(f"{i}. {opp}")

                with tab3:
                    st.markdown("**Similar Words:**")
                    for i, sim in enumerate(info.similar_words, 1):
                        st.write(f"{i}. {sim}")

                with tab4:
                    st.markdown("**Example Sentences:**")
                    for i, sent in enumerate(info.sentences, 1):
                        st.write(f"{i}. {sent}")

            except Exception as e:
                st.error("Oops! Something went wrong. Please try again.")
                logging.exception(f"Error processing word '{word}': {e}")

# ------------------ Word History ------------------
if st.session_state.history:
    st.markdown("📚 ***Your Word History***")
    for w, data in st.session_state.history.items():
        with st.expander(f"🔤 {w.capitalize()}", width="stretch"):
            st.write(f"**Meaning:** {data['meaning']}")
            st.write(f"**Opposites:** {', '.join(data['opposites'])}")
            st.write(f"**Similar Words:** {', '.join(data['similar_words'])}")
            st.write("**Sentences:**")
            for s in data["sentences"]:
                st.write(f"- {s}")
