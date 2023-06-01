import os
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
import torch
import torch.nn.functional as F
from podder.utils import Podcast, Speaker, remove_repeated_sequences, is_exe_installed
import time
from soxan.models import Wav2Vec2ForSpeechClassification
import logging

class Podder:
    def __init__(self,
                 link,
                 name,
                 auth_token,
                 start_time = None,
                 end_time = None,
                 min_speakers = None,
                 max_speakers = None,
                 num_speakers = None):
              
        self.link = link
        self.name = name
        self.auth_token = auth_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.num_speakers = num_speakers
        
        if start_time is not None:
            self.start_time = start_time * 1000
        else:
            self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time * 1000
        else:
            self.end_time = end_time

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Verify install of ffmpeg and yt-dlp
        if is_exe_installed('ffmpeg') is False:
            raise FileNotFoundError('ffmpeg not found. Please install ffmpeg and add it to your PATH.\n Visit https://ffmpeg.org/download.html for more information.')
        if is_exe_installed('yt-dlp') is False:
            raise FileNotFoundError('yt-dlp not found. Please install yt-dlp and add it to your PATH.\n Visit https://github.com/yt-dlp/yt-dlp for more information.')

    
    def process_podcast(self) -> Podcast:
        
        pod = Podcast()
        pod.link = self.link    
        pod.name = self.name

        # start execution timer
        timer_start = time.time()

        # Load the model for text classification
        from transformers import pipeline
        classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k = None)
        
        # Download the audio from the link using yt-dlp
        os.system(f'yt-dlp --quiet -o {self.name} -x --audio-format "wav" {self.link}')
        #TODO: require global ffmpeg location and yt-dlp location. Handle those errors appropriately
        
        # see if cuda is available for processing, if not use cpu and warn the user that it may be slow
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            msg = "cuda not detected as an available compute platform.\n This may take a long time."
            self.logger.warning(msg)
            print(msg)
            #TODO add some time estimates for cpu vs gpu
        
        pod.process_device = device.type
        
        audio = AudioSegment.from_wav(self.name + '.wav')
        
        # segment the downloaded audio if we are not using the full podcast
        if audio.duration_seconds > self.end_time/1000:
            audio = audio[self.start_time:self.end_time]
            trimmed_audio_name = f'{self.name}_{self.start_time/1000}-{self.end_time/1000}s'
            audio.export(trimmed_audio_name + '.wav', format="wav")
            pod.end_time = audio.duration_seconds
        else:
            trimmed_audio_name = self.name
            pod.end_time = self.end_time/1000

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                            use_auth_token=self.auth_token).to(device)
        
        # when the auth token is invalid, the pipeline is None so raise an error
        if pipeline is None:
            raise ValueError("Invalid auth token provided for pyannote/speaker-diarization")
        
        # run the pipeline with the appropriate parameters
        if (self.min_speakers is not None
            and self.max_speakers is not None
            and self.num_speakers is None):
            
            diarization = pipeline(f'{trimmed_audio_name}.wav', min_speakers=self.min_speakers, max_speakers=self.max_speakers)
        
        elif (self.min_speakers is None
                and self.max_speakers is None
                and self.num_speakers is not None):
            diarization = pipeline(f'{trimmed_audio_name}.wav', num_speakers=self.num_speakers)
        
        else:
            diarization = pipeline(f'{trimmed_audio_name}.wav')


        trimmed_audio = [0]*len(diarization._labels)

        speaker_list = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            try:
                pod.speakers[speaker]
            except:
                pod.add_speaker(Speaker(speaker))
                speaker_list.append(speaker)
            
            pod.speakers[speaker].add_time(turn.start, turn.end)

            start_time = turn.start * 1000  # Start is at 5th second: 5000 ms
            end_time =  turn.end * 1000   # End is at 10th second: 10000 ms

            if trimmed_audio[speaker_list.index(speaker)] == 0:
                trimmed_audio[speaker_list.index(speaker)] = audio[start_time:end_time]
            else:
                trimmed_audio[speaker_list.index(speaker)] = trimmed_audio[speaker_list.index(speaker)] + audio[start_time:end_time]

        for ind, speaker in enumerate(speaker_list):
            speaker_filename = f'{trimmed_audio_name}_{speaker}.wav'
            trimmed_audio[ind].export(speaker_filename, format="wav")

        # delete the audio and diarization objects to free up memory
        del trimmed_audio
        del audio
        del diarization
        del pipeline

        # Load the model for audio classification
        config = AutoConfig.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")
        model_audio = Wav2Vec2ForSpeechClassification.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition").to(device)

        # Load the model for text generation
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model_text = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
        model_text.config.forced_decoder_ids = None
        sampling_rate = 16000

        # loop through each speaker's audio and process it
        for speaker in speaker_list:
            speaker_filename = f'{trimmed_audio_name}_{speaker}.wav'
            speech_array, _sampling_rate = torchaudio.load(speaker_filename)

            # resample the audio to a frequency our audio model can handle
            resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
            speech = resampler(speech_array[1]).squeeze().numpy()

            # split the audio into 10 second chunks and process each chunk
            for i in range(0, len(speech), sampling_rate*10):

                # if the last chunk is less than 10 seconds, skip it. Short chunks can cause errors
                if len(speech) > i + sampling_rate*10:
                    input_features = processor(speech[i:i+sampling_rate*10-1], sampling_rate=sampling_rate, return_tensors="pt").input_features 
                else:
                    continue
                
                # run our 10 seconds of audio through the model to get a transcription   
                predicted_ids = model_text.generate(input_features.to(device))
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

                # run our 10 seconds of audio through the model to get an emotion classification
                inputs = feature_extractor(speech[i:i+sampling_rate*10-1], sampling_rate=sampling_rate, return_tensors="pt", padding=True)
                inputs = {key: inputs[key].to(device) for key in inputs}

                with torch.no_grad():
                    logits = model_audio(**inputs).logits

                # run the logits through a softmax to get the probabilities
                scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

                # create a list of dictionaries with the emotion and score
                audio_emotions = [{"Emotion": config.id2label[i], "Score": score} for i, score in
                        enumerate(scores)]

                # run the transcription through the text classification model
                text_emotions = classifier(transcription)[0]
                
                transcription = remove_repeated_sequences(transcription[0])
                pod.speakers[speaker].add_transcript(transcription)

                # add the emotion classifications to the speaker object for both audio and text
                for emotion in text_emotions:
                    pod.speakers[speaker].add_text_emotion(emotion['label'], emotion['score'])

                for emotion in audio_emotions:
                    pod.speakers[speaker].add_audio_emotion(emotion['Emotion'], emotion['Score'])

            # normalize the emotions for each speaker now that we know we are through all clips
            pod.speakers[speaker].normalize_text_emotions()
            pod.speakers[speaker].normalize_audio_emotions()

            # get the most common words for each speaker from their transcripts and store them in the speaker object
            pod.speakers[speaker].most_common_words(num_words=10)

        # end the execution timer and store the execution time in the podcast object
        timer_end = time.time()
        execution_time = timer_end - timer_start
        pod.execution_time = execution_time

        return pod
