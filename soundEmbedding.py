import re
import numpy as np
import torch
from sound2vec import prepare_model, segment_audio, prepare_feature_extractor, three_second


def main():
  model_path = "facebook/wav2vec2-large-xlsr-53"

  model = prepare_model(model_path)

  for wav_file in files:
      count = 0
      name = re.search('(\d+)\.', wav_file).group().replace('.', '_')
      audio_input, sample_rate = segment_audio(wav_file)
      feature_extractor = prepare_feature_extractor(sample_rate)

      if len(audio_input) > 1:
          for audio in audio_input:
              input_values = feature_extractor(audio, return_tensors="pt").input_values
              logits = model(input_values).logits

              output = logits.detach().numpy()
              np.save(f'{name}{count}.npy', output)

              count += 1
      else:
          input_values = feature_extractor(audio_input, return_tensors="pt").input_values
          logits = model(input_values).logits

          output = logits.detach().numpy()
          np.save(f'{name}.npy', output)

# Only first three second segements
#   for wav_file in files:
#       name = re.search('(\d+)\.', wav_file).group().replace('.', '_')

#       audio_input, sample_rate = three_second(wav_file)
#       feature_extractor = prepare_feature_extractor(sample_rate)

#       input_values = feature_extractor(audio_input, return_tensors="pt").input_values
#       logits = model(input_values).logits

#       output = logits.detach().numpy()
#       np.save(f'{name}.npy', output)
          
if __name__ == "__main__":
    main()
