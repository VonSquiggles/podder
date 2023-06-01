# podder
A python library that diarizes a youtube podcast and analyzes each speaker's emotion by processing both their speech and the transcription of what they said. 

I leveraged the following models to accomplish this:

1. [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) - used to break up podcast audio by each speaker
2. [openai/whisper-base](https://huggingface.co/openai/whisper-base) - used to transcribe podcast audio
3. [harshit345/xlsr-wav2vec-speech-emotion-recognition](https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition) - used to analyze speaker sentiment/emotion based on individual speaker's audio
4. [bhadresh-savani/distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) - used to analyze speaker sentiment/emotion based on individual speaker's text transcript

# Installation Instructions >=1.0.0:
> :warning: I developed this on a Windows laptop with an Nvidia GPU. Installation instructions will look different for Mac or Linux

1. Install pytorch. I recommend going to https://pytorch.org/ to find the right install command for your device. For me, this was
        ```
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
        ```
> :warning: I encourage installing torch this way so you can leverage your GPU for inference (if you have one)
2. Install develop branch of pyannote-audio. Latest pyannote-audio version at time of readme update (v2.1.1) does not support latest torch version (>=2.0.0), thus must use dev branch
        ```
        pip install -qq https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip
        ```
3. Install podder (eventually will get this up on pypi)
        ```
        pip install https://github.com/VonSquiggles/podder/archive/refs/heads/main.zip
        ```
4. Install my fork of soxan (in my fork I took https://github.com/m3hrdadfi/soxan and formatted it as an installable package)
        ```
        pip install https://github.com/VonSquiggles/soxan/archive/refs/tags/v0.1.0.tar.gz
        ```
5. Download https://github.com/yt-dlp/yt-dlp and add it to PATH so it can be called from anywhere on your device.
   1. This allows you to download youtube videos as .wav files onto your device.
   
6. Download ffmpeg (a dependency of yt-dlp) and add it to PATH so it can be called from anywhere on your device
   1. I went to https://www.gyan.dev/ffmpeg/builds/ and downloaded `ffmpeg-release-essentials.zip`
   
7. Sign up for a https://huggingface.co/ account (free). 
   
8.  Accept user conditions for https://hf.co/pyannote/speaker-diarization
    
9.  Acceupt user conditions for https://hf.co/pyannote/segmentation 
    
10. Get a User Access Token at https://hf.co/settings/tokens.


# Example

Now try it out with the first couple minutes of your favorite podcast!:
```
>> from podder import Podder

>> podcast = PodcastSpeakers(link="YOUTUBE_PODCAST_LINK",
                        name="PODCAST_NAME", # this is just what your .wav files will be named with
                        auth_token="HUGGINGFACE_AUTH_TOKEN",
                        start_time=0,
                        end_time=120)

>> pod_data = podcast.process_podcast()
>> print(pod_data)

Podcast: podcast_name
Speakers: {'SPEAKER_00':
Times: [Clip: 0.4s - 4.6s, Clip: 11.6s - 29.4s]
Transcript:  Hey hi here is some fake transcript with fake audio and text emotion weights
Audio emotions: {'happiness': 0.2, 'disgust': 0.3, 'anger': 0.1, 'fear': 0.2, 'sadness': 0.2}
Text emotions: {'joy': 0.1, 'anger': 0.4, 'fear': 0.1, 'sadness': 0.2, 'surprise': 0.1, 'love': 0.1}
Words: [('a', 4), ('for', 2), ('been', 2), ('getting', 2), ('incredible', 2)]
, 'SPEAKER_01':
Times: [Clip: 4.7s - 11.0s, Clip: 18.2s - 18.4s]
Transcript: hey another fake transcript and more fake weights. enjoy!
Audio emotions: {'happiness': 0.2, 'disgust': 0.3, 'anger': 0.1, 'fear': 0.2, 'sadness': 0.2}
Text emotions: {'joy': 0.1, 'anger': 0.4, 'fear': 0.1, 'sadness': 0.2, 'surprise': 0.1, 'love': 0.1}
Words: [('it', 2), ('as', 2), ('i', 2), ('to', 2), ('do', 2)]
}
Length: 0
Start time: 0
End time: 30.0
Execution time: 26.73457670211792

```

I have a Dell XPS 9510 with a 7th gen Intel i7 processor, 16GB RAM, and a Nvidia RTX 3050

I tested the example on 5 minutes of a podcast with 4 speakers and recorded the time it took using my GPU vs my CPU:

* GPU (`device = 'cuda'`) took 67 seconds

* CPU (`device = 'cpu'`) took 463 seconds
