from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from transformers import AutoFeatureExtractor, AutoModel


feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
model = AutoModel.from_pretrained("microsoft/wavlm-large")

stim_dir='/om/user/ehoseini/fmri_DNN/AnnSentSet1_stims/stimuli_alignment_handfix/'

import torch
from transformers import WavLMConfig, WavLMModel
from transformers import Wav2Vec2Processor, WavLMForCTC
# Initializing a WavLM facebook/wavlm-base-960h style configuration
configuration = WavLMConfig()
# Initializing a model (with random weights) from the facebook/wavlm-base-960h style configuration
model = WavLMModel(configuration)
# Accessing the model configuration
configuration = model.config

from transformers import Wav2Vec2Processor, WavLMModel
import torch
from datasets import load_dataset, load_from_disk
from datasets import load_dataset, Audio


dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = load_from_disk("/om/user/ehoseini/MyData/fmri_DNN/outputs/AnnSentSet1_sentences_dataset/")

sampling_rate = dataset.features["audio"].sampling_rate

processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")


model

with torch.no_grad():
    outputs = model(**inputs,output_hidden_states=True)

[x.shape for x in outputs.hidden_states]

model = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
transcrip_pair=[]
for sample in tqdm(dataset):
    inputs = processor(sample["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs,output_hidden_states=True)
    predicted_ids = torch.argmax(outputs['logits'], dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    transcrip_pair.append([transcription,sample['text']])
