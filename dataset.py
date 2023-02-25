from datasets import load_dataset, Audio
import subprocess
import os
from pathlib import Path
from pydub import AudioSegment

def load_data():
	ds = load_dataset('google/MusicCaps', split='train')
	return ds

def load_batch(ds, sample_start, batch_size):
	cores = 4                 # How many processes to use for the loading
	writer_batch_size = 10000  # How many examples to keep in memory per worker. Reduce if OOM.

	# Just select some samples 
	samples = ds.select(range(sample_start, sample_start + batch_size))
        
	samples = samples.map(
			process,
			num_proc=cores,
			writer_batch_size=writer_batch_size,
			keep_in_memory=False
		)# .cast_column('audio', Audio(sampling_rate=sampling_rate))
        
	return samples


def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='/tmp/musiccaps',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False

    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format mp3 -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" {url_base}{video_identifier}
    """.strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'

def process(example):
    try:
        data_dir = "./music_data/" # Where to save the data
        outfile_path = str(data_dir + f"{example['ytid']}.mp3")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, log = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )
            song = AudioSegment.from_mp3(outfile_path)
            ten_seconds = 10 * 1000
            last_10_seconds = song[-ten_seconds:]
            last_10_seconds.export(outfile_path, format="mp3")
            
        example['audio'] = outfile_path
        example['download_status'] = status
    except Exception as e:
         example["audio"] = ""
         example["download_status"] = False
    return example

def main(start_batch = 0):
    ds = load_data()
    # Download all audio data
    batch_size = 32
    num_batches = (len(ds) // batch_size) + (0 if (len(ds) % batch_size) == 0 else 1)

    failed_batch_indices = []

    for i in range(start_batch, num_batches):
        print(f"Batch {i}")
        print(i * batch_size)
        try:
            samples = load_batch(ds, i * batch_size, batch_size)
            samples.to_csv(f"./samples/samples_{i}.csv")
        except Exception as e:
             print(e)
             print(f"Failed Batch {i}")
             failed_batch_indices.append(i)
    print("Failed Batches")
    print(failed_batch_indices)
        
if __name__ == "__main__":
     main()
