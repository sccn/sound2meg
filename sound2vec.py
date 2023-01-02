from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import librosa
import fairseq
import torch
import pandas as pd


# make 3 second segments in audio file
def segment_audio(audio_file):
    audio_input, sample_rate = librosa.load(audio_file)
    audio_segment = sample_rate * 3
    segments = []
    if len(audio_input) >= audio_segment * 2:
      check = True
      while check:
          segments.append(audio_input[:audio_segment])
          audio_input = audio_input[audio_segment:]
          if len(audio_input) < audio_segment:
              check = False
    else:
      segments.append(audio_input[:audio_segment])
    return segments, sample_rate

# return first 3 second segment in audio file
def three_second(audio_file):
    audio_input, sample_rate = librosa.load(audio_file)
    audio_segment = sample_rate * 3

    segment = audio_input[:audio_segment]
    return segment, sample_rate


def read_event(tsv_path):
    event = pd.read_csv("sub-A2014_task-auditory_events.tsv", sep = '\t')
    event = event[event['type'] == 'Sound']

    batch["onset"], batch["sample"], batch["audio_file"] = event['onset'], event['sample'], event['audio_file']
    
    return batch


def prepare_dataset(batch, feature_extractor):
    for audio in audio_file_directory:
        if audio in batch["audio_file"]:
            segment, sample_rate = three_second(audio)
            #segment, sample_rate = segment_audio(audio)
            
            feature_extractor = prepare_feature_extractor(sample_rate)
            batch["input_values"] = feature_extractor(segment, return_tensors="pt").input_values

    return batch

def prepare_model(model_path):
    model = Wav2Vec2ForCTC.from_pretrained(model_path)

    return model

def prepare_feature_extractor(sampling_rate):
    # check that all files have the correct sampling rate
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=sampling_rate, padding_value=0.0, do_normalize=True, return_attention_mask=True)

    return feature_extractor
