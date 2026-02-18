# Traf Web Crawler 🌐

**Traf** is a Streamlit-based web application that extracts main content from web pages or subtitles from YouTube videos. It utilizes Google Gemini AI to translate, summarize, and enable chat-based interaction with the extracted content.

## ✨ Key Features

*   **Web Crawling**: Extracts main text content from a given URL using `trafilatura`.
*   **YouTube Transcript Extraction**: Automatically retrieves subtitles (transcripts) when a YouTube video URL is provided.
*   **AI Translation & Summary**: Translates extracted text into Korean or summarizes key points using Google Gemini models.
*   **AI Chatbot**: Engage in context-aware conversations with Gemini based on the crawled content.
*   **PDF Download**: Download original, translated, summarized text, or chatbot history as styled PDF files using `weasyprint`.
*   **Markdown Viewer**: Cleanly displays text in Markdown format with support for table formatting.

## 🛠️ Installation

This project recommends running in a Python 3.11+ environment.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Homin0321/traf.git
    cd traf
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    # venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

You need to set up an API key to use the Google Gemini API.

1.  Create a `.env` file in the project root directory.
2.  Enter your API key as follows:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```
    (You can obtain an API key from Google AI Studio.)

## 🚀 Usage

Run the Streamlit app with the following command:

```bash
streamlit run app.py
```

The browser will open automatically, and you can access the application at `http://localhost:8501`.

## 📂 Project Structure

*   `app.py`: Main application logic (UI rendering, session management, Gemini integration, etc.).
*   `utils.py`: Utility functions for fixing Markdown rendering issues.
*   `styled.html`: HTML/CSS template used for PDF generation.
*   `requirements.txt`: List of required Python libraries.

## 💡 Tips

*   Select a **Gemini Model** from the sidebar to choose the model that fits your performance or speed needs.
*   Inputting a YouTube link instead of a web page will automatically retrieve the transcript.
*   Change the **View Mode** to switch between Crawled, Translated, Summary, and Chatbot views.
*   If you like the result, click the **Download PDF** button in the sidebar to save it.
