import os
from pydub import AudioSegment

READ_FILES = '.'
SAVE_FILES = "./wav_files"
BATCH_SIZE = 100


def flac_to_wav():
    os.makedirs(SAVE_FILES, exist_ok=True)

    flac_files = [f for f in os.listdir(READ_FILES) if f.endswith(".flac")]

    for i in range(0, len(flac_files), BATCH_SIZE):
        batch_files = flac_files[i:i + BATCH_SIZE]
        for filename in batch_files:
            flac_path = os.path.join(READ_FILES, filename)
            wav_filename = filename.replace(".flac", ".wav")
            wav_path = os.path.join(SAVE_FILES, wav_filename)
            
            audio = AudioSegment.from_file(flac_path, format="flac")
            audio.export(wav_path, format="wav")
            print(f"Converted {flac_path} to {wav_path}")

if __name__ == '__main__':
    flac_to_wav()
