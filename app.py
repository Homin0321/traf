import os
import re

import markdown
import streamlit as st
import trafilatura
from dotenv import load_dotenv
from google import genai
from weasyprint import HTML
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from utils import fix_markdown_symbol_issue

# --- 1. Constants ---
MODEL_OPTIONS = [
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
# Session state keys
SESSION_KEYS = {
    "crawled_text": "crawled_text",
    "translated_text": "translated_text",
    "summary_text": "summary_text",
    "url_to_crawl": "url_to_crawl",
    "selected_model": "selected_model",  # Added for model selection
}

# Gemini prompts
PROMPTS = {
    "summary": """
        **System Instruction:** You are an expert content summarizer. Your goal is to analyze the provided text and generate a structured, high-quality summary in Korean.

        **Task:**
        1.  **Analyze** the text to identify the core message and key details.
        2.  **Generate** a summary in Korean that includes:
            -   **One-Line Summary:** A single sentence capturing the essence of the content.
            -   **Key Takeaways:** A bulleted list of the most critical points (3-5 items).
            -   **Detailed Summary:** A well-structured paragraph explaining the context and details.

        **Formatting:**
        -   Use Markdown headers (##, ###) and bullet points.
        -   Ensure the tone is professional and objective.

        **Text to Summarize:**
        """,
    "translation": """
        **System Instruction:** You are a professional translator fluent in both English and Korean. Your goal is to provide a natural, high-quality translation of the given text into Korean.

        **Task:**
        1.  **Translate** the text into Korean.
        2.  **Maintain** the original tone, nuance, and formatting (e.g., Markdown headers, lists).
        3.  **Ensure** the Korean is idiomatic and grammatically correct, avoiding literal translations that sound unnatural.

        **Output Format:**
        -   Return only the translated text in Markdown format.

        **Text to Translate:**
        """,
}


# --- 3. Core Logic Functions ---
def initialize_session_state():
    """Initializes session state."""
    for key in SESSION_KEYS.values():
        if key not in st.session_state:
            if key == SESSION_KEYS["selected_model"]:
                st.session_state[key] = MODEL_OPTIONS[0]
            else:
                st.session_state[key] = ""

    if "chat_display_history" not in st.session_state:
        st.session_state["chat_display_history"] = []


def is_valid_url(url):
    """A simple function to validate a URL."""
    regex = re.compile(
        r"^(?:http|ftp)s?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, url) is not None


def get_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_youtube_transcript(video_id):
    """Fetches the transcript of a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Try to fetch English transcript first
            transcript = transcript_list.find_transcript(["en"])
        except NoTranscriptFound:
            # Fallback: take the first available transcript
            transcript = next(iter(transcript_list))

        return " ".join([item.text for item in transcript.fetch()])

    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None


def format_markdown_tables(text):
    """
    Formats markdown tables by inserting a header separator line.
    It identifies tables by looking for consecutive lines with the same number of '|' characters (at least 2).
    """
    lines = text.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        pipe_count = line.count("|")

        if pipe_count >= 2:
            table_lines_indices = [i]
            j = i + 1
            while j < len(lines) and lines[j].count("|") == pipe_count:
                table_lines_indices.append(j)
                j += 1

            if len(table_lines_indices) > 1:
                # This block is a table.
                header_line = lines[i]

                header_stripped = header_line.strip()
                cols = header_stripped.split("|")
                if header_stripped.startswith("|"):
                    cols = cols[1:]
                if header_stripped.endswith("|"):
                    cols = cols[:-1]
                num_cols = len(cols)

                if num_cols > 0:
                    separator = "|" + " --- |" * num_cols

                    new_lines.append(header_line)
                    new_lines.append(separator)

                    for k in range(1, len(table_lines_indices)):
                        new_lines.append(lines[table_lines_indices[k]])

                    i = j
                    continue

        # If not a table or a single line, just add the line
        new_lines.append(line)
        i += 1

    return "\n".join(new_lines)


def crawl_url(url):
    """Crawls a URL and saves the result to session_state using trafilatura."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        metadata = trafilatura.extract_metadata(downloaded)

        result = trafilatura.extract(
            downloaded,
            include_links=True,
            include_tables=True,
            include_formatting=True,
            output_format="markdown",
        )
        if result:
            if metadata:
                result = f"""### {metadata.title}
- {metadata.author}, {metadata.date}
- {metadata.description}
---

{result}"""
            st.session_state[SESSION_KEYS["crawled_text"]] = result
        else:
            st.error("Failed to retrieve markdown content from the crawl result.")
            st.session_state[SESSION_KEYS["crawled_text"]] = ""
    else:
        st.error("Failed to retrieve content from the URL.")
        st.session_state[SESSION_KEYS["crawled_text"]] = ""

    # Reset LLM text and summary after crawling
    st.session_state[SESSION_KEYS["translated_text"]] = ""
    st.session_state[SESSION_KEYS["summary_text"]] = ""

    if "chat_session" in st.session_state:
        del st.session_state["chat_session"]
    if "chat_display_history" in st.session_state:
        st.session_state["chat_display_history"] = []


def get_gemini_key():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Please check your .env file.")
        return None
    return GEMINI_API_KEY


@st.cache_resource
def get_gemini_client():
    GEMINI_API_KEY = get_gemini_key()
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


def get_response_text(response):
    """
    Extracts text from Gemini response, filtering out non-text parts to avoid warnings.
    """
    if not response:
        return ""

    # Check if we have candidates and parts to manually extract text
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            if text_parts:
                return "".join(text_parts)

    # Fallback to .text if manual extraction fails
    if hasattr(response, "text"):
        return response.text

    return ""


@st.cache_data
def convert_by_gemini(instruction, text):
    """Converts text using Gemini."""
    client = get_gemini_client()
    if not client:
        return None

    selected_model = st.session_state.get(
        SESSION_KEYS["selected_model"], MODEL_OPTIONS[0]
    )

    try:
        response = client.models.generate_content(
            model=selected_model, contents=instruction + text
        )
        if not response:
            st.error("Gemini returned no valid response.")
            return None

        text = get_response_text(response)
        if not text:
            st.error("Gemini returned no text in response.")
            return None

        return fix_markdown_symbol_issue(text.strip())
    except Exception as e:
        st.error(f"An error occurred during Gemini processing: {e}")
        return None


def initialize_chat_session(context):
    """
    Initialize or reset chat session with transcript context.
    Args:
        context (str): The transcript text to provide context for the chat
    """
    try:
        # Create new chat session with Gemini
        st.session_state["chat_session"] = get_gemini_chat(context)
        st.session_state["chat_display_history"] = []
    except Exception as e:
        st.error(f"Failed to initialize Gemini chat: {e}")
        # Reset session state on error
        st.session_state["chat_session"] = None
        st.session_state["chat_display_history"] = []


def get_gemini_chat(context):
    client = get_gemini_client()
    if not client:
        return None

    selected_model = st.session_state.get(
        SESSION_KEYS["selected_model"], MODEL_OPTIONS[0]
    )

    chat = client.chats.create(model=selected_model, history=[])
    # Initialize with transcript context
    chat.send_message(
        f"""
        **System Instruction:** You are a helpful AI assistant tasked with answering questions based on the provided text.

        **Context:**
        {context}

        **Task:**
        -   Answer user questions accurately using *only* the information from the provided context.
        -   If the answer is not found in the context, politely state that you cannot find the information in the provided text.
        -   Answer in Korean unless the user asks in another language.
        """
    )
    return chat


def chat_with_gemini(context):
    """
    Implements chat interface for transcript Q&A using Gemini API.
    Args:
        context (str): The transcript text to chat about
    """
    # Initialize chat session if it doesn't exist
    if (
        "chat_session" not in st.session_state
        or st.session_state["chat_session"] is None
    ):
        initialize_chat_session(context)

    # Create chat input interface
    user_input = st.chat_input("Ask something about the video...")

    # Process user message and get AI response
    if user_input:
        # Add user message to chat history
        st.session_state["chat_display_history"].append(
            {"role": "user", "content": user_input}
        )
        try:
            # Get AI response
            response = st.session_state["chat_session"].send_message(user_input)
            answer = fix_markdown_symbol_issue(get_response_text(response).strip())
            # Add AI response to chat history
            st.session_state["chat_display_history"].append(
                {"role": "assistant", "content": answer}
            )
        except Exception as e:
            st.error(f"Gemini Q&A Error: {e}")
            # Reset chat session on error
            initialize_chat_session(context)
            return

    # Display chat messages
    for message in st.session_state["chat_display_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def download_pdf(markdown_text):
    """Converts Markdown to PDF and returns the PDF data."""
    if not markdown_text:
        return None

    # Convert markdown to HTML
    html_text = markdown.markdown(
        markdown_text, extensions=["extra", "codehilite", "tables", "fenced_code"]
    )

    # Read the HTML template from file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "styled.html")
    with open(file_path, "r", encoding="utf-8") as f:
        styled_html_template = f.read()

    # Replace the placeholder with the actual content
    styled_html = styled_html_template.replace("{{content}}", html_text)

    return HTML(string=styled_html).write_pdf()


@st.dialog(title="Markdown Code", width="large")
def show_markdown_code(markdown_text):
    st.code(markdown_text, language="markdown")


# --- 4. UI Rendering Functions ---
def render_sidebar():
    """Renders the sidebar UI."""
    st.sidebar.title("Traf Web Crawler 🌐")

    st.sidebar.selectbox(
        "Select Gemini Model",
        MODEL_OPTIONS,
        key=SESSION_KEYS["selected_model"],
    )

    st.sidebar.text_input("Enter the URL to crawl", key=SESSION_KEYS["url_to_crawl"])

    if st.sidebar.button("Crawl", width="stretch"):
        url = st.session_state[SESSION_KEYS["url_to_crawl"]]
        if not url:
            st.sidebar.warning("Please enter a URL.")
            return

        if not is_valid_url(url):
            st.sidebar.error("Please enter a valid URL (e.g., https://example.com)")
            return

        video_id = get_video_id(url)
        if video_id:
            with st.spinner(f"Fetching transcript for {url}..."):
                transcript = get_youtube_transcript(video_id)
                if transcript:
                    st.session_state[SESSION_KEYS["crawled_text"]] = transcript
                    # Reset LLM text and summary after crawling
                    st.session_state[SESSION_KEYS["translated_text"]] = ""
                    st.session_state[SESSION_KEYS["summary_text"]] = ""

                    if "chat_session" in st.session_state:
                        del st.session_state["chat_session"]
                    if "chat_display_history" in st.session_state:
                        st.session_state["chat_display_history"] = []
                else:
                    st.error(
                        "Failed to retrieve transcript. The video might not have captions or they are disabled."
                    )
        else:
            with st.spinner(f"Crawling {url}..."):
                try:
                    crawl_url(url)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    st.sidebar.checkbox("Fix Table Format", key="format_markdown")

    st.sidebar.radio(
        "View Mode",
        ["Crawled", "Translated", "Summary", "Chatbot"],
        key="view_mode",
    )

    if st.sidebar.button("Show Markdown Code", width="stretch"):
        mode = st.session_state.get("view_mode", "Crawled")
        text_to_show = ""
        if mode == "Crawled":
            text_to_show = st.session_state.get(SESSION_KEYS["crawled_text"])

        elif mode == "Translated":
            text_to_show = st.session_state.get(SESSION_KEYS["translated_text"])
        elif mode == "Summary":
            text_to_show = st.session_state.get(SESSION_KEYS["summary_text"])
        elif mode == "Chatbot":
            chat_history = st.session_state.get("chat_display_history", [])
            assistant_messages = [
                m["content"] for m in chat_history if m["role"] == "assistant"
            ]
            if assistant_messages:
                text_to_show = assistant_messages[-1]
            else:
                st.sidebar.warning("No assistant answer to show.")
                text_to_show = None
        else:
            text_to_show = None

        if text_to_show:
            show_markdown_code(text_to_show)
        elif mode != "Chatbot":
            st.sidebar.warning(f"No content in '{mode}' to show.")

    mode = st.session_state.get("view_mode", "Crawled")
    text_to_download = ""
    file_name = "download.pdf"

    if mode == "Crawled":
        text_to_download = st.session_state.get(SESSION_KEYS["crawled_text"])
        file_name = "crawled.pdf"

    elif mode == "Translated":
        text_to_download = st.session_state.get(SESSION_KEYS["translated_text"])
        file_name = "translated.pdf"
    elif mode == "Summary":
        text_to_download = st.session_state.get(SESSION_KEYS["summary_text"])
        file_name = "summary.pdf"
    elif mode == "Chatbot":
        chat_history = st.session_state.get("chat_display_history", [])
        assistant_messages = [
            m["content"] for m in chat_history if m["role"] == "assistant"
        ]
        if assistant_messages:
            text_to_download = assistant_messages[-1]
            file_name = "chatbot_answer.pdf"
        else:
            text_to_download = None
    else:
        text_to_download = None

    if text_to_download:
        pdf_data = download_pdf(text_to_download)
        if pdf_data:
            st.sidebar.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=file_name,
                mime="application/pdf",
                width="stretch",
            )

    st.sidebar.divider()
    st.sidebar.caption("Powered by [trafilatura](https://github.com/adbar/trafilatura)")


def render_main_content():
    """Renders the main content area."""
    mode = st.session_state.get("view_mode", "Crawled")

    crawled_text = st.session_state.get(SESSION_KEYS["crawled_text"])
    translated_text = st.session_state.get(SESSION_KEYS["translated_text"])
    summary_text = st.session_state.get(SESSION_KEYS["summary_text"])

    if mode == "Crawled":
        if crawled_text:
            if st.session_state.get("format_markdown"):
                crawled_text = format_markdown_tables(crawled_text)
            st.markdown(crawled_text)
        else:
            st.info("No crawled content. Enter a URL and click the 'Crawl' button.")

    elif mode == "Translated":
        if crawled_text and not translated_text:
            with st.spinner("Translating content with Gemini..."):
                translated_text = convert_by_gemini(
                    PROMPTS["translation"], crawled_text
                )
                st.session_state[SESSION_KEYS["translated_text"]] = translated_text
                st.session_state[SESSION_KEYS["summary_text"]] = (
                    ""  # Reset summary to force re-summarization
                )

        if translated_text:
            st.markdown(translated_text)
        else:
            st.info("No translated content.")

    elif mode == "Summary":
        if crawled_text and not summary_text:
            with st.spinner("Generating summary with Gemini..."):
                summary_text = convert_by_gemini(PROMPTS["summary"], crawled_text)
                st.session_state[SESSION_KEYS["summary_text"]] = summary_text

        if summary_text:
            st.markdown(summary_text)
        else:
            st.info("No summarized content.")

    elif mode == "Chatbot":
        if crawled_text:
            chat_with_gemini(crawled_text)
        else:
            st.info("No crawled content to chat about. Please run the crawler first.")


# --- 5. Main Application Execution ---
def main():
    """Main application function."""
    st.set_page_config(layout="wide", page_title="Traf Web Crawler", page_icon="🌐")

    # Load .env file
    load_dotenv()

    # Initialize session state
    initialize_session_state()

    # Render UI
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
