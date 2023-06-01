import subprocess
from collections import Counter
import re

class Clip:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
    
    def to_dict(self):
        return {'start': self.start, 'stop': self.stop}
    
    def __str__(self) -> str:
        return f"Clip: {self.start}s - {self.stop}s"
    
    def __repr__(self) -> str:
        return self.__str__()

class Podcast:
    def __init__(self):
        self.speakers = {}
        self.process_device = ''
        self.start_time = 0
        self.end_time = 0
        self.length = 0
        self.link = ''
        self.name = ''
        self.execution_time = 0

    def add_speaker(self, speaker):
        if isinstance(speaker, Speaker):
            self.speakers[speaker.name] = speaker
        else:
            print("Invalid speaker object")

    def get_speaker(self, name):
        return self.speakers.get(name, None)

    def to_dict(self):
        return {'link': self.link, 'name': self.name, 'length': self.length, 'start_time': self.start_time, 'end_time': self.end_time, 'execution_time': self.execution_time, 'speakers':
            {name: speaker.to_dict() for name, speaker in self.speakers.items()}}
    
    def __str__(self):
        return f"Podcast: {self.name}\nSpeakers: {self.speakers}\nLength: {self.length}\nStart time: {self.start_time}\nEnd time: {self.end_time}\nExecution time: {self.execution_time}"

class Speaker:
    def __init__(self, name):
        self.name = name
        self.times = []
        self.transcript = ""
        self.audio_emotions = {
            "happiness": 0,
            "disgust": 0,
            "anger": 0,
            "fear": 0,
            "sadness": 0,
        }
        self.text_emotions = {
            "joy": 0,
            "anger": 0,
            "fear": 0,
            "sadness": 0,
            "surprise": 0,
            "love": 0,
        }
        self.words = []

    def add_time(self, start, stop):
        self.times.append(Clip(start, stop))

    def add_text_emotion(self, emotion, value):
        try:
            self.text_emotions[emotion] += value
        except KeyError:
            print(f"Invalid emotion: {emotion}")

    def add_audio_emotion(self, emotion, value):
        try:
            self.audio_emotions[emotion] += value
        except KeyError:
            print(f"Invalid emotion: {emotion}")

    def add_transcript(self, transcript):
        self.transcript += transcript
    
    def normalize_text_emotions(self):
        total = sum(self.text_emotions.values())
        if total != 0:  # avoid division by zero
            for emotion in self.text_emotions:
                self.text_emotions[emotion] /= total
    
    def normalize_audio_emotions(self):
        total = sum(self.audio_emotions.values())
        if total != 0:  # avoid division by zero
            for emotion in self.audio_emotions:
                self.audio_emotions[emotion] /= total
    
    def to_dict(self):
        return {
            "times": [time.to_dict() for time in self.times],
            "transcript": self.transcript,
            "audio_emotions": self.audio_emotions,
            "text_emotions": self.text_emotions,
            "words": self.words
        }
    
    def most_common_words(self, num_words=50):
        # Remove punctuation from the transcript
        transcript = re.sub(r'[^\w\s]', '', self.transcript)
        words = transcript.lower().split()
        word_counts = Counter(words)
        common_words = word_counts.most_common(num_words)
        self.words = common_words
    
    def __str__(self) -> str:
        return f"\nTimes: {self.times}\nTranscript: {self.transcript}\nAudio emotions: {self.audio_emotions}\nText emotions: {self.text_emotions}\nWords: {self.words}\n"
    
    def __repr__(self) -> str:
        return self.__str__()

def remove_repeated_sequences(text):
    # Split the text into sequences based on comma and full stop
    sequences = re.split('[,.] ?', text)
    
    seen = set()
    result = []
    rslt = text
    for sequence in sequences:
        # If the sequence has not been seen before, add it to the result
        if sequence not in seen:
            seen.add(sequence)
    
    for one_seen in seen:
        rslt = rslt.replace(one_seen, '#here', 1)
        rslt = rslt.replace(one_seen, '')
        rslt = rslt.replace('#here', one_seen)
    
    rslt = rslt.replace(' ,', '')
    rslt = rslt.replace(' .', '')
    # Join the result back into a single string
    return rslt

def is_exe_installed(exe_name):
    try:
        subprocess.run([exe_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False
