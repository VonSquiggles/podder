from podder import Podder
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()  # take environment variables from .env.
USE_AUTH_TOKEN = os.getenv('USE_AUTH_TOKEN')

yt_pod = Podder(link='https://www.youtube.com/watch?v=C762HWSz67w',
               name="podcast_name",
               auth_token=USE_AUTH_TOKEN,
               start_time=0,
               end_time= 30)

rslt = yt_pod.process_podcast()

print(f'\nExecution time: {rslt.execution_time} seconds\n Execution device: {rslt.process_device}\n')

# Convert podcast to a dictionary
podcast_dict = rslt.to_dict()

# Serialize to JSON
json_str = json.dumps(podcast_dict)

now = datetime.now()
timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')

dir_path = './podcasts'

# Create the directory if it doesn't already exist
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

file_path = os.path.join(dir_path, f'{rslt.name}_speaker_data_{timestamp}.json')

with open(file_path, 'w') as f:
    f.write(json_str)