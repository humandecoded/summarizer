import whisper
import requests
import json
import os
import argparse
from dotenv import load_dotenv
import privatebinapi
from datetime import datetime

# this is a script that will read in a list of files
# transcribe them using openai-whisper, save that file
# pass that text over to llama3:80b to summarize
# save that summary and upload it to privatebin
# 

# function that takes a large block of text and returns a list of smaller chunks
def chunk_string_by_words(text, chunk_size):
    words = text.split()  # Split the text into words
    chunks = []
    for i in range(0, len(words), chunk_size):
        # Join words to form chunks of 'chunk_size'
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

#function that takes a file path to an audio file and returns the transcription
def WhisperTranscribe(audio_file):
    transcription = whisper.load_model("small.en").transcribe(audio_file)
    return transcription["text"]

# function that takes a transcription and returns a summarized version of the text
def LlamaSummarize(text, prompt="summarize this: "):
    #start with just first 2000 words of text
    #if len(text.split()) > 2500:
     #   text = " ".join(text.split()[:2500])
    
    #set up the post request to the llama3 api
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt + text,
        "stream": False,
        "model": "llama3:70b-instruct",
        "keep_alive": 0
    }
    
    #make the post request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    #need to check the response to see if llm has hallucinated slashes or short response
    if len(json.loads(response.text)["response"]) < 200 or "\\" in json.loads(response.text)["response"]:
        print("Response was too short or hallucinated slashes. Trying again with a different model")
        print(f"Sample of text: {text[:100]}")
        data = {
            "prompt": prompt + text,
            "stream": False,
            "model": "llama3:70b-instruct",
            "keep_alive": 0
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
    # return the response text
    return json.loads(response.text)["response"]
    
def main():
    #use argparse to get the path to the audio files
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-list", type=str, required=True, help="list of files to summarize")
    parser.add_argument("-p", "--paste", action="store_true", help="paste the summary to privatebin")
    parser.add_argument("-l", "--log-location", type=str, help="location to save the log file.", default="")
    args = parser.parse_args()
    
    load_dotenv()

    #get the logfile path
    if args.log_location != "":
        log_location = parser.log_location
    else:
        log_location = os.getcwd()


    #define the path to a folder of audio files
    audio_file_path = args.file_list
    if audio_file_path.endswith('/'):
        audio_file_path = audio_file_path[:-1]

    #read in file names from the list
    file_list =  []
    with open(audio_file_path, "r") as f:
        for line in f.readlines():
            file_list.append(line.strip())
            f.flush()

    # set up the log file
    log_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".log"

    # open the log file
    with open(f"{log_location}/{log_name}", "w") as log_file:
            #loop through each file
        for audio_file_path in file_list:
            # extract just the filename for audio_file_path
            # pull the last string to the right of a slash
            audio_file_name = audio_file_path.split("/")[-1]

            #transcribe the audio
            print(f"Transcribing {audio_file_path}")
            transcription = WhisperTranscribe(audio_file_path)
            
            # chunk out the text and get string of summaries
            print("Breaking text into chunks and summarizing chunks")
            transcription_list = chunk_string_by_words(transcription, 2500)
            # summarize each chunk
            summary_string = ""
            prompt = "you are a summarizer of podcasts and videos. This is a sample of a larger episode. Summarize with two paragraphs: "
            for transcription_chunk in transcription_list:
                print(prompt)
                summary = LlamaSummarize(transcription_chunk, prompt=prompt )
                summary_string = summary_string + summary + "\n"

            print(f"Started with {len(transcription.split())} words, ended with {len(summary_string.split())} words")
            #summarize the summary
            print(f"Summarizing {audio_file_path}")
            prompt = "You are a summarizer of podcasts and videos. This text represents the summaries of different sections of an episode. Create a bullet pointed list pointing out the highlights of the summaries: "
            print(prompt)
            summary = LlamaSummarize(summary_string, prompt=prompt)

            print(summary + "\n")

            #paste the summary to privatebin
            if args.paste:
                response = privatebinapi.send(os.getenv("PRIVATEBIN_URL"), text=summary)
                log_file.write(f"{audio_file_path}\n {response['full_url']}\n\n")
                log_file.write("--------------------------------------------------\n\n\n")
                
            else:
                log_file.write(f"{audio_file_path}\n {summary}\n\n")
                log_file.write("--------------------------------------------------\n\n\n")
                log_file.flush()
                


if __name__ == '__main__':
	main()

    