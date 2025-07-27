import streamlit as st
import moviepy.editor as mp
from moviepy.editor import *
from pytube import YouTube
import os
import torch
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


BASE_PATH = '.'

def get_transcript(audio_file,language = 'en'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  #If you have GPU else it will use cpu
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-small"    # define the model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # create the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=32,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,)

    result = pipe(audio_file, return_timestamps=True, generate_kwargs = {"task" : "transcribe",  "language" : f"<|{language}|>"})

    result = result["chunks"]
    return result


def format_lyrics(lyrics):
    formatted_lyrics = ""
    for line in lyrics:
        text = line["text"]
        formatted_lyrics += f"{text}\n"

    # creatting path for the text file
    TEXT_PATH = BASE_PATH + "/lyrics"

    # checking if it does not exist create it and leave it if it exists
    if not os.path.exists(TEXT_PATH):
      os.makedirs(TEXT_PATH)

    # saving the text file
    with open(TEXT_PATH + "/lyrics.txt", "w") as file:
        file.write(formatted_lyrics)

    return formatted_lyrics.strip('\n')


def extract_audio_to_file(video_path, audio_path):
    """
    Uses the moviepy package to extract and write the entire audio content
    from a video to a new file. Handles BrokenPipeError with retries.
    """
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)


def display_text_from_file(uploaded_file):
    """
    Reads text from an uploaded file and displays it in a scrollable area with markdown formatting.

    Args:
        uploaded_file (streamlit.uploadedfile.UploadedFile): The uploaded file object.

    Returns:
        None
    """

    try:
        # Combine text display and markdown formatting in a single step
        st.write('<style>.scrollable-text-container {overflow-y: scroll; max-height: 600px;  border:/'
                 '1px solid #cccccc;padding:10px; border-radius: 5px; color: rgb(0,150,0); background-color: #f5f5f5;}</style>',unsafe_allow_html=True)
        st.markdown(f'<div class="scrollable-text-container">{uploaded_file}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error reading file: {e}")


def download_video(url, output_path='./video'):
    # Use pytube to download the video
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    print(video)

    # Download the video to a file
    out_file = video.download(output_path=output_path)

    # Set up new filename for the MP3
    new_file = out_file.replace(".mp4", ".mp3")
    # Use moviepy to convert video audio to mp3
    video_clip = AudioFileClip(out_file)
    video_clip.write_audiofile(new_file)

    # Remove the original video file
    os.remove(out_file)

    return new_file  # Return the path to the new file

def convert_to_mp3(file_path):
    audio = AudioSegment.from_file(file_path, format=file_path.split('.')[-1])
    file_name = os.path.splitext(file_path)[0]
    audio.export(f"{file_name}.mp3", format="mp3")

def main():

    AUDIO_BASE = f"{BASE_PATH}/audio"
    VIDEO_BASE = f"{BASE_PATH}/video"

    if not os.path.exists(AUDIO_BASE):
        os.makedirs(AUDIO_BASE)

    if not os.path.exists(VIDEO_BASE):
        os.makedirs(VIDEO_BASE)

    st.set_page_config(page_title="AudioChat Transcriber", layout="wide")
    # App title and introduction
    st.title("AudioChat Transcriber")
    st.write("Enter the path to your file or paste the text directly below:")

    with st.container():  # Wrap elements in a container for styling

        # File input method options (flexbox)
        col1, col2 = st.columns(2)  # Create two columns for better layout
        with col1:
            text_upload_method = st.radio("File Input Method:", ("Audio", "Video", "Youtube link","Tweeter Space Record Link"), key="file_method")
        with col2:
            # Language selection (flexbox)
            language = st.radio("Select the expected language of the audio:",
                                ("English", "Spanish", "French", "Deutch"), key="language")


    dic_language = {
        'English':'en',
        'Spanish':'sp',
        'French':'fr',
        'Deutch':'de'
    }

    langInshort = dic_language[language]

    css = """
       <style>
       .download-button {
           display: inline-block;
           padding: 10px 20px;
           background-color: rgb(0,200,0);
           color: white !important;
           text-align: center;
           text-decoration: none;
           font-size: 16px;
           border-radius: 5px;
           margin-top: 20px;
       }
       </style>
    """

    if text_upload_method == "Audio":
        uploaded_file = st.file_uploader("Choose a text file:", type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac','opus'])
        if uploaded_file is not None:
            basename = uploaded_file.name

            if basename.rsplit('.', 1)[1] == 'mp3':
                # getting the name of the filename
                filename = basename.rsplit('.', 1)[0]
                audio_path = f"{AUDIO_BASE}/{filename}.mp3"

                # Save the file to the current directory
                with open(f'{audio_path}', 'wb') as file:
                    file.write(uploaded_file.read())

                # Getting
                lyrics = get_transcript(audio_path,language=langInshort)
                lyrics = format_lyrics(lyrics)
                print(lyrics)
                display_text_from_file(lyrics)

                os.remove(audio_path)

                # Add a styled download button
                st.markdown(
                    f'<a href="data:text/plain;charset=utf-8,{lyrics}" download="transcribed_text.txt" class="download-button">Download</a>',
                    unsafe_allow_html=True)

                # Add the CSS style
                st.markdown(css, unsafe_allow_html=True)

            else:

                audio_path = f"{AUDIO_BASE}/{basename}"

                # Creatinng new path with the mp3 extension
                filename = basename.rsplit('.', 1)[0]
                new_path = f"{AUDIO_BASE}/{filename}.mp3"

                # Save the file to the current directory
                with open(f'{audio_path}', 'wb') as file:
                    file.write(uploaded_file.read())

                convert_to_mp3(audio_path)
                print(f'Finished conversion to .mp3 of:{basename}')
                os.remove(audio_path)

                # Transcribing the new audio file
                lyrics = get_transcript(new_path, language=langInshort)
                print(lyrics)
                lyrics = format_lyrics(lyrics)
                display_text_from_file(lyrics)
                os.remove(new_path)

                # Add a styled download button
                st.markdown(
                    f'<a href="data:text/plain;charset=utf-8,{lyrics}" download="transcribed_text.txt" class="download-button">Download</a>',
                    unsafe_allow_html=True)

                # Add the CSS style
                st.markdown(css, unsafe_allow_html=True)


    elif text_upload_method == "Video":

        uploaded_file = st.file_uploader("Choose a video:", type=["mp4"])
        if uploaded_file is not None:
            basename = uploaded_file.name

            if basename.rsplit('.', 1)[1] == 'mp4':
                # getting the name of the filename
                filename = basename.rsplit('.', 1)[0]
                video_path = f"{VIDEO_BASE}/{filename}.mp4"
                audio_path = f"{AUDIO_BASE}/{filename}.mp3"

                with open(f'{video_path}', 'wb') as file:
                    file.write(uploaded_file.read())

                extract_audio_to_file(video_path, audio_path)

                lyrics = get_transcript(audio_path, language=langInshort)
                lyrics = format_lyrics(lyrics)

                display_text_from_file(lyrics)

                # Add a styled download button
                st.markdown(
                    f'<a href="data:text/plain;charset=utf-8,{lyrics}" download="transcribed_text.txt" class="download-button">Download</a>',
                    unsafe_allow_html=True)

                # Add the CSS style
                st.markdown(css, unsafe_allow_html=True)



    else:
        url = st.text_input("Enter your text here:")
        file_name = ''
        if url:

            try:

                file_name = download_video(url, AUDIO_BASE)
                lyrics = get_transcript(file_name,language=langInshort)
                lyrics = format_lyrics(lyrics)
                display_text_from_file(lyrics)
                # Add a styled download button
                st.markdown(
                    f'<a href="data:text/plain;charset=utf-8,{lyrics}" download="transcribed_text.txt" class="download-button">Download</a>',
                    unsafe_allow_html=True)

                # Add the CSS style
                st.markdown(css, unsafe_allow_html=True)

            except Exception as e:
                print(e)
                st.write('Bad Url or network problems: Copy a valid URL or Verify the network')


if __name__ == "__main__":
    main()
