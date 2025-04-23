import os
import time
import requests
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

transcription_value = ""


def get_assemblyai_api_key():
    return os.getenv("ASSEMBLYAI_API_KEY")


def transcribe_speech_with_assemblyai(filepath):
    if filepath is None:
        return "No audio found, please retry."

    api_key = get_assemblyai_api_key()
    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    print(f"Uploading audio file at {filepath} to AssemblyAI")

    with open(filepath, "rb") as audio_file:
        response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"authorization": api_key},
            data=audio_file
        )

    upload_url = response.json()["upload_url"]

    json_data = {"audio_url": upload_url}
    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=json_data,
        headers=headers
    )

    transcript_id = response.json()["id"]
    transcription_status_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        response = requests.get(transcription_status_url, headers=headers)
        transcript = response.json()

        if transcript["status"] == "completed":
            print("Transcription completed successfully")
            return transcript["text"]

        print("Transcription in progress... waiting")
        time.sleep(3)


def transcribe_speech(filepath):
    return transcribe_speech_with_assemblyai(filepath)


def store_transcription(output):
    global transcription_value
    transcription_value = output
    return output


def setup_gradio_interface():
    mic_transcribe = gr.Interface(
        fn=lambda x: store_transcription(transcribe_speech(x)),
        inputs=gr.Audio(sources="microphone", type="filepath"),
        outputs=gr.Textbox(label="Transcription")
    )

    file_transcribe = gr.Interface(
        fn=lambda x: store_transcription(transcribe_speech(x)),
        inputs=gr.Audio(type="filepath"),
        outputs=gr.Textbox(label="Transcription")
    )

    interface = gr.Blocks()
    with interface:
        gr.TabbedInterface(
            [mic_transcribe, file_transcribe],
            ["Transcribe Microphone", "Transcribe Audio File"]
        )

    return interface


if __name__ == "__main__":
    if not get_assemblyai_api_key():
        print("WARNING: ASSEMBLYAI_API_KEY not found in .env file!")
        exit(1)

    interface = setup_gradio_interface()

    print("Starting transcription interface...")
    print("Access the interface at: http://127.0.0.1:8000")
    interface.launch(
        share=False,
        server_port=8000,
        prevent_thread_lock=True
    )

    input("Press Enter to stop the server and exit...")

    interface.close()

    print("Transcription result:")
    print(transcription_value)
