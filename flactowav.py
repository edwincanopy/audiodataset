import os
from pydub import AudioSegment

READ_FILES = '.'
SAVE_FILES = "./wav_files"

def flac_to_wav():

    for filename in os.listdir(READ_FILES):

        if filename.endswith(".flac"):

            flac_path = os.path.join(READ_FILES, filename)
            
            wav_filename = filename.replace(".flac", ".wav")
            wav_path = os.path.join(SAVE_FILES, wav_filename)
            
            audio = AudioSegment.from_file(flac_path, format="flac")
            audio.export(wav_path, format="wav")
            print(f"Converted {flac_path} to {wav_path}")

if __name__ == '__main__':
    flac_to_wav()