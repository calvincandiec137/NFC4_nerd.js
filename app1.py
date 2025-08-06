import streamlit as st
import os
from modules.run_splitter import main
from pathlib import Path
from modules.response import res_main
import time
import shutil

# Configure the page
st.set_page_config(page_title="Document Processor", layout="wide")
st.title("📄 Doc.AI")

# Initialize session state
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_path" not in st.session_state:
    st.session_state.file_path = ""

# Custom CSS for better appearance
st.markdown("""
<style>
    .upload-box {
        padding: 30px;
        border: 2px dashed #ccc;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .file-info {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .chat-message {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

folder_path= "C:/Users/Nitesh/OneDrive/Desktop/NFC4_nerd.js/database"

code_option = st.radio(
    "Choose the code version:",
    ("Previously Available Code", "Newly Trained Code")
)

def empty_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # delete file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # delete directory
            except Exception as e:
                st.error(f"[❌] Failed to delete {file_path}: {e}")
        st.success(f"✅ Folder '{folder_path}' has been emptied.")
    else:
        st.warning(f"⚠️ Folder '{folder_path}' does not exist.")

# Set default preprocessed file path when using "Previously Available Code"
if code_option == "Previously Available Code":
    st.session_state.document_processed = True
    st.session_state.file_path = "C:\\Users\\Nitesh\\OneDrive\\Desktop\\NFC4_nerd.js\\statictemp\\aepyornis-island.pdf"  # Adjust to your actual preprocessed file
    # Skip the upload section and go straight to chatbot
    st.session_state.messages=[]  # This will refresh the page and skip the upload section

# --------------------------- Upload Section (ONLY for Newly Trained Code) --------------------------- #
if code_option == "Newly Trained Code":
    with st.container():
        st.subheader("Upload Document for Processing")
        with st.form("upload_form"):
            uploaded_file = st.file_uploader(
                "Choose a file (PDF, DOCX, TXT, MD)",
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=False,
                key="file-uploader",
                help="Maximum file size: 16MB"
            )
            process_btn = st.form_submit_button("Process Document")

        if process_btn and uploaded_file is not None:
            try:
                empty_folder(folder_path)
                database_dir = Path("database")
                save_path = database_dir / uploaded_file.name
                database_dir.mkdir(parents=True, exist_ok=True)

                with st.spinner(f"Saving and processing {uploaded_file.name}..."):
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.session_state.messages = []

                    with st.expander("📄 File Details", expanded=True):
                        cols = st.columns(3)
                        cols[0].metric("File Name", uploaded_file.name)
                        cols[1].metric("File Type", uploaded_file.type)
                        cols[2].metric("File Size", f"{len(uploaded_file.getvalue())/1024:.1f} KB")
                        st.success(f"File successfully saved to:\n`{save_path}`")

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for percent in range(0, 101, 10):
                        time.sleep(0.1)
                        progress_bar.progress(percent)
                        status_text.text(f"Processing... {percent}%")

                    main(uploaded_file.name)

                    progress_bar.empty()
                    status_text.empty()

                    st.success("✅ Document processing completed successfully!")
                    st.session_state.document_processed = True
                    st.session_state.file_path = str(save_path)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Please check the console for more details")
                print(f"Error processing file: {e}")

        elif process_btn and uploaded_file is None:
            st.warning("⚠️ Please upload a file first.")

# --------------------------- Chatbot Section (shown when document is processed) --------------------------- #
if st.session_state.document_processed:
    st.markdown("---")
    st.subheader("💬 Document Chatbot")

    # Display chat messages using simple st.write
    for message in st.session_state.messages:
        role = "You" if message["role"] == "user" else "Assistant"
        st.write(f"**{role}:** {message['content']}")  # Use Markdown for bold labels
        # conversation+=f"{role}: {message['content']}+\n"
    # Chat input
    query = st.chat_input("Ask your question about the document...")

    # with open("conversation_log.txt", "w", encoding="utf-8") as f:
    #     f.write(conversation)

    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            try:
                # Get response from RAG system
                answer = res_main(query)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Refresh to show new messages
                st.rerun()

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
