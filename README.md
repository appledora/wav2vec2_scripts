## Wav2Vec2 for Low Resource Speech Recognition

### Install conda environment from file
```bash
conda env create -f environment.yml
conda activate lstt
```
### Data Preparation 
Audio files must be in wav format and sampled to 16kHz. The audio files must be in the following directory structure:
```
BASE_PATH
├── train_files_dir
│   ├── file1.wav
│   ├── file2.wav
│   ├── ...
│   ├── metadata.csv
├── test_files_dir
│   ├── file1.wav
│   ├── file2.wav
│   ├── ...
│   ├── metadata.csv
.....
```
#### Sampling audio files to 16kHz
```python
AUDIO_PATH = 'path/to/audio/files'
RESAMPLED_PATH = 'path/to/resampled/audio/files'
AUDIO_FILES = os.listdir(AUDIO_PATH)
from utils import resample_audio
for file in AUDIO_FILES:
    resample_audio(sample_rate=16000, input_file_path=os.path.join(AUDIO_PATH, file), output_dir=RESAMPLED_PATH)
```

#### The metadata.csv file
This files has only two columns: `file_name` and `transcript`. The `file_name` column contains the name of the audio file and the `transcripts` column contains the transcript of the audio file. A sample metadata file is available in the [`sample csv`](./sample%20csv) folder.

### Fine-tuning Wav2Vec2
Use the following script to start fine-tuning:
```bash
python3 ft.py --train_dir path/to/train/files --valid_dir path/to/valid/files --audio_dir base/directory/containing/split/folders --repo_name hfrepo/you/want/to/finetune/from --generate_vocab
```
#### Arguments
The training arguments are available in the `args.json` file. To change the run parameters, edit this file.


### To run inference 
Inference is the same as fine-tuning command:
```bash
python inference.py --test_dir path/to/test/files --audio_dir base/directory/containing/split/folders --ckpt_dir your/saved/checkpoint/dir --results_dir path/to/save/results
```

