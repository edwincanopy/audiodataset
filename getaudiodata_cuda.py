import numpy as np
import random
import os
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import detect_silence
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import Dataset
from tqdm import tqdm
import sys

# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# ---
AUDIO_FOLDER = sys.argv[1]  # contains wav files
OUTPUT_FOLDER = 'clips'  # contains split audio clips and their transcriptions
BATCH_SIZE = 5

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")
model.to(device)
# ---

def split_wav(folder, min_silence_len=500, silence_thresh=-50):
    wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    wav_files.sort()

    wav_data = []
    
    for wav_file in wav_files:
        file_path = os.path.join(folder, wav_file)
        
        audio_segment = AudioSegment.from_wav(file_path)
        
        segment_length = 10 * 1000  # milliseconds
        duration_ms = len(audio_segment)
        start = 0
        i = 1
        rand_num = random.randrange(100000, 999999)

        while start < duration_ms:
            end = start + segment_length
            
            segment_silences = detect_silence(
                audio_segment[start:end],
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )
            
            if segment_silences:
                silence_start, silence_end = segment_silences[0]
                end = start + silence_end
            
            segment_data = audio_segment[start:end]
  
            segment_data = segment_data.set_frame_rate(16000)
            
            output_file_name = f"{rand_num}{wav_file[:-4]}{i}.wav"
            output_file_path = os.path.join(OUTPUT_FOLDER, output_file_name)
            
            segment_data.export(output_file_path, format="wav")
            
            start = end
            i += 1

def save_audio_data(audio_paths):
    transcriptions = []
    audio_inputs = []
    
    for audio_path in audio_paths:
        # Load the audio and move it to the GPU
        audio_input, sample_rate = torchaudio.load(audio_path)
        audio_input = audio_input.to(device)
        
        audio_inputs.append({
            'array': audio_input.squeeze().cpu().numpy(),  # Keep data on the CPU for processing
            'sampling_rate': sample_rate
        })
    
    # Prepare the input features and attention mask, moving them to the GPU
    inputs = processor(
        [audio['array'] for audio in audio_inputs],
        return_tensors="pt",
        sampling_rate=16000,
        padding=True
    )
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    # Set language to English
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    with torch.no_grad():
        if attention_mask is not None:
            ids = model.generate(input_features, attention_mask=attention_mask, forced_decoder_ids=forced_decoder_ids)
        else:
            ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

        transcriptions = processor.batch_decode(ids, skip_special_tokens=True)
    
    return audio_inputs, [speech.strip() for speech in transcriptions]

def main():
    split_wav(AUDIO_FOLDER)

    dataset_dict = {'audio': [], 'text': []}

    split_audio = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.wav')]
    split_audio.sort()

    # Use tqdm to display the progress bar
    for i in tqdm(range(0, len(split_audio), BATCH_SIZE), desc="Processing audio batches"):
        batch = split_audio[i:i+BATCH_SIZE]
        batch = [os.path.join(OUTPUT_FOLDER, f) for f in batch]

        audio, speech = save_audio_data(batch)
        print([a['array'].shape for a in audio])
        print(speech)

        dataset_dict['audio'].extend(audio)
        dataset_dict['text'].extend(speech)

    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk('audio-dataset')


if __name__ == '__main__':
    main()
