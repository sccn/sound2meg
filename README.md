# sound2meg

Original paper
https://arxiv.org/pdf/2208.12266.pdf

Mel paper
https://arxiv.org/abs/2006.11477

Wav2vec model and paper
- Model https://huggingface.co/facebook/wav2vec2-large-xlsr-53
- Paper https://arxiv.org/abs/2006.13979

Code on Colab
https://colab.research.google.com/github/sccn/sound2meg/blob/main/Spatial_Attention.ipynb

# Other papers

## Mapping brain data (fMRI) with latent space of GPT-2 - we could do that with EEG and MEG
https://www.nature.com/articles/s41562-022-01516-2?utm_content=animation

## Assessing if self-supervised learning (learning to detect neighboring EEG segment) improve classification performance 
https://arxiv.org/abs/2007.16104v1

## Training on massive datasets for transfer learning in EEG (right?)
https://www.frontiersin.org/articles/10.3389/fnhum.2021.653659/full
Arno: We could do this for our large corpus of child data (3000 subjects)

## Correlating latent space of stable diffusion and fMRI (we would do it with EEG)
https://sites.google.com/view/stablediffusion-with-brain/?s=09

## Diffusion and fMRI
https://mind-vis.github.io
Abdu: This is similar to the stable diffusion one that I came across a while back. It seems to be using a more complicated model, but it also used fMRI.

## MEG classification
https://hal.inria.fr/hal-03808304
Abdu: I haven't looked into this in detail but it seems to be some network encoding MEG signals for better classification (I'm guessing kind of like wav2vec but for brain data?). The code seems to be open source at https://github.com/facebookresearch/deepmeg-recurrent-encoder so we can experiment with some new ideas. 

