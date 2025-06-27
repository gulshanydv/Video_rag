# app.py
import streamlit as st
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
from moviepy import VideoFileClip
import whisper
import tempfile

# --- Load Env ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Transcription Function ---
def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# --- Handle Uploaded Videos ---
def handle_uploaded_videos(videos, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    transcripts = []

    for uploaded_file in videos:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_file.read())
            tmp_video_path = tmp_video.name

        video_name = Path(uploaded_file.name).stem
        audio_path = os.path.join(output_dir, f"{video_name}.mp3")
        transcript_path = os.path.join(output_dir, f"{video_name}_transcript.txt")

        extract_audio(tmp_video_path, audio_path)
        transcript = transcribe_audio(audio_path)

        with open(transcript_path, "w") as f:
            f.write(transcript)

        transcripts.append((video_name, transcript))
        st.success(f"Transcribed: {video_name}")

    return transcripts

# --- Load and Chunk Transcripts ---
def load_vectorstore(transcript_folder="./output"):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for path in Path(transcript_folder).glob("*.txt"):
        with open(path, "r") as f:
            text = f.read()
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"source": path.stem}))
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# --- Initialize RAG Chatbot ---
def init_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
    vectorstore = load_vectorstore()
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True
    )

# --- Streamlit UI ---
st.set_page_config(page_title="Video Transcript QA", layout="wide")
st.title(" Video Transcript Chatbot")

# --- Upload UI ---
st.sidebar.header("Upload Video Files")
uploaded_videos = st.sidebar.file_uploader("Upload one or more .mp4 files", accept_multiple_files=True, type=["mp4"])
if st.sidebar.button("Transcribe Videos"):
    if uploaded_videos:
        handle_uploaded_videos(uploaded_videos)
        st.session_state.qa_chain = init_chain()  # Refresh vector store
    else:
        st.sidebar.warning("Please upload at least one video file.")

# --- Init Chain + Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = init_chain()

# --- Question Input ---
question = st.text_input("Ask a question about your videos:")

if question:
    with st.spinner("Thinking..."):
        result = st.session_state.qa_chain({
            "question": question,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append((question, result["answer"]))
        # st.markdown(f"**AI:** {result['answer']}")

# --- Show Chat History ---
if st.session_state.chat_history:
    st.markdown("---")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
