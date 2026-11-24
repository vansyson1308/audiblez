# Audiblez: Generate  audiobooks from e-books

[![Installing via pip and running](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml)
[![Git clone and run](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/audiblez)
![PyPI - Version](https://img.shields.io/pypi/v/audiblez)

### v5: Russian and 10 languages with Qwen3-TTS, YouTube-ready output, automatic quality control

Audiblez generates `.m4b` audiobooks (and `.mp4` videos for YouTube) from e-books (`.epub`, `.fb2`) and story
scripts (`.txt`, `.md`, `.docx`).

Two speech engines are available:

| Engine | Languages | Strengths |
|---|---|---|
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | 🇺🇸 🇬🇧 🇪🇸 🇫🇷 🇮🇳 🇮🇹 🇯🇵 🇧🇷 🇨🇳 | very fast, runs well on CPU |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0) | 🇷🇺 🇬🇧 🇩🇪 🇫🇷 🇪🇸 🇮🇹 🇵🇹 🇨🇳 🇯🇵 🇰🇷 | very natural and expressive, voice cloning and voice design |

The engine is chosen automatically from the e-book language (Kokoro when it supports the language, Qwen3-TTS otherwise,
e.g. for Russian), or with `--engine`.

What happens between the e-book and the audio file matters as much as the voice:

- **Any input**: EPUB, FB2 (`.fb2`, `.fb2.zip`, the usual Russian e-book format) and story scripts: plain text
  (one paragraph per line, or hard-wrapped text that is re-joined; UTF-8 or Windows-1251), Markdown and Word/Google
  Docs `.docx`. A short first line becomes the title, `Глава 1` / `I` / `## ...` lines become chapters, `* * *`
  scene breaks, `- Привет, - сказала она` dialogue hyphens become dashes.
- **Clean text**: block-level EPUB extraction in reading (spine) order, no footnote markers or footnote sections,
  no table of contents / copyright pages, `<br>` and soft hyphens handled, books stored as a single file are split
  into their real chapters (`I`, `II`, `Глава 5`...), real chapter titles in the `.m4b` and in YouTube timestamps.
- **Russian text normalization**: numbers in the right case and gender (`в 1812 году` → *в тысяча восемьсот
  двенадцатом году*, `до 5 часов` → *до пяти часов*, `Пётр I` → *Пётр Первый*, `XIX в.` → *девятнадцатый век*),
  dates, money, percentages, abbreviations (`т. е.`, `и т. д.`, `г-н`), restoration of `ё` (`еще` → `ещё`, otherwise
  it is mispronounced), ALL-CAPS titles (`ОН НЕ ОЖИДАЛ…`) read as normal words while abbreviations (МВД, СССР)
  stay. Use `--lexicon` for names and words you want pronounced differently, and `--dry-run` to read the
  normalized script before synthesizing.
- **Russian word stress** (`--stress ruaccent`): [RUAccent](https://github.com/Den4ikAI/ruaccent) marks the stressed
  vowel of every word from its context (за́мок / замо́к, все / всё) and the text is read by
  [a Qwen3-TTS fine-tune that follows the marks](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf)
  (Apache-2.0; its model card warns that part of its training texts has no declared license).
- **Natural pacing**: the text is synthesized by sentences grouped within paragraphs (never across them), the silence
  the model leaves around each chunk is trimmed, and controlled pauses are inserted between sentences, paragraphs,
  headings, scene breaks (`* * *`) and chapters. `--pause-scale` makes the whole book calmer or faster. The pauses
  are filled with the voice's own background noise ("room tone", measured in each chapter, about -65 dBFS for
  Qwen3-TTS) instead of digital silence, so the background does not switch off between sentences
  (`--no-room-tone` to disable).
- **Automatic quality control** (Qwen3-TTS): neural TTS sometimes skips words or hallucinates on long texts. Every
  chunk is checked (speaking rate, length limit, silence, clipping and, if `faster-whisper` is installed, a Whisper
  transcription compared with the text) and regenerated with another seed when it fails. The chunks that never pass
  are listed in `qa_report.txt` so you can listen to them.
- **Mastering**: two-pass EBU R128 loudness normalization (-16 LUFS, -1.5 dBTP), 48 kHz, a single lossy encoding.
- **Resumable**: every synthesized chunk is cached (atomically); an interrupted run continues where it stopped,
  changing one paragraph only re-synthesizes that paragraph, and a fully cached run loads no model at all.
  `--time-limit 11.5` stops cleanly before a cloud session ends.
- **Fast on GPUs**: every visible GPU is used in parallel (one worker process each, e.g. the free 2 x T4 of
  Kaggle: `kaggle/audiblez_kaggle.ipynb`), and the optional [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
  backend (CUDA graphs, `--qwen-backend fast`) runs the same model ~2x (T4) to ~5x (RTX 4090) faster per GPU.
  `audiblez-bench` measures backend x precision x batch size on your GPU and recommends the fastest setting that
  keeps the quality.
- **YouTube**: `--youtube` creates an `.mp4` (cover over a blurred background, 1080p) and a text file with the chapter
  timestamps for the description; `--episode-hours 2` splits the book into parts of about 2 hours (at chapter
  boundaries, or at paragraph boundaries for very long chapters).

![Audiblez GUI on MacOSX](./imgs/mac.png)

On a Google Colab's T4 GPU via Cuda, **Kokoro takes about 5 minutes to convert "Animal's Farm" by Orwell**
(about 160,000 characters), at a rate of about 600 characters per second. On an M2 MacBook Pro CPU it takes about
1 hour. Qwen3-TTS (1.7B parameters) is much heavier: a CUDA GPU is strongly recommended. Measured on a 4-core
Xeon CPU: voice cloning runs about 8x slower than real time (1 hour of audio takes about 8 hours). Published
single-GPU figures for the 1.7B model (faster-qwen3-tts author): 0.45x real time on a Tesla T4 with qwen-tts,
0.93x with CUDA graphs; 4.2x on an RTX 4090. With the two T4 of Kaggle that is about 1.1 h for a 2 h episode.


## Russian audiobooks with Qwen3-TTS (quick start)

```bash
sudo apt install ffmpeg sox                          # on Ubuntu/Debian (brew install ffmpeg sox on Mac)
pip install "audiblez[asr]"                          # [asr] adds the automatic Whisper quality control

audiblez book.epub --dry-run                         # 1. check chapters and the normalized text (script.txt)
audiblez book.epub --chapters 1                      # 2. listen to the first chapter
audiblez book.epub --youtube --episode-hours 2       # 3. the whole book as ~2 h YouTube videos + .m4b
audiblez story.txt --youtube --stress ruaccent       # a story script, with automatic word stress
audiblez-bench                                       # fastest settings for this GPU
```

Step-by-step guide (Vietnamese): [docs/huong-dan-audiobook-tieng-nga.md](docs/huong-dan-audiobook-tieng-nga.md);
free GPU rendering on Kaggle: [kaggle/README.md](kaggle/README.md).

Without voice options, Russian books are read by the bundled `ru_male_narrator` voice. Other bundled voices:
`audiblez-voice list`. For the best result, design (or record) your own narrator and use it for the whole book:

```bash
audiblez-voice design --preset ru_male_narrator -n 6          # 6 candidates in ./voices/ru_male_narrator/
audiblez-voice design --name grandpa --language ru \
    --instruct "An older male voice, 65 years old, native Russian, husky, warm, unhurried storyteller"
audiblez-voice create --audio my_voice.wav --text "Exact transcript of the recording" -o voices/me
audiblez-voice preview --voice-profile voices/ru_male_narrator/candidate_03 -o test.wav
audiblez book.epub --voice-profile voices/ru_male_narrator/candidate_03 --youtube --episode-hours 2
```

Each candidate prints its pitch (a deep baritone is around 90-120 Hz) and Whisper error: the voice design model does
not always follow the description, so listen and keep the one you like. A voice profile is a folder with a 6-15 s
clean reference (`reference.flac`) and its exact transcript (`reference.txt`). Why cloning: the built-in Qwen3-TTS
speakers (`--speaker Ryan`...) speak Russian with a foreign accent, and `--instruct` voice design alone can drift
between sentences; a cloned reference keeps the same native voice for hours. Only clone voices you have the rights to.

Useful options for long books: `--batch-size 8` (more GPU throughput on large GPUs; default 4 on CUDA), `--lexicon names.txt` (lines like
`Германн = Герман`), `--temperature 0.7` (steadier, less expressive), `--retry-failed` (regenerate the chunks of the
QA report), `--loudness -14` (louder masters), `--cover image.jpg` (video/m4b artwork).

## How to install the Command Line tool

If you have Python 3.10-3.13 on your computer, you can install it with pip (Kokoro is only installed on
Python 3.10-3.12; on 3.13, as on Kaggle and Colab, Qwen3-TTS is used).
You also need `espeak-ng` (Kokoro) and `ffmpeg` installed on your machine:

```bash
sudo apt install ffmpeg espeak-ng                   # on Ubuntu/Debian 🐧
pip install audiblez
```

```bash
brew install ffmpeg espeak-ng                       # on Mac 🍏
pip install audiblez
```

Then you can convert an .epub directly with:

```
audiblez book.epub -v af_sky
```

It creates a `book.audiblez/` working folder (synthesis cache, chapter wavs, normalized script, QA report) and at the
end a `book.m4b` file with the whole book and its chapters, that you can listen with VLC or any audiobook player.
It will only produce the `.m4b` file if you have `ffmpeg` installed on your machine.

## How to run the GUI

The GUI is a simple graphical interface to use audiblez.
You need some extra dependencies to run the GUI:

```
sudo apt install ffmpeg espeak-ng 
sudo apt install libgtk-3-dev        # just for Ubuntu/Debian 🐧, Windows/Mac don't need this
  
pip install audiblez pillow wxpython
```

Then you can run the GUI with:
```
audiblez-ui
```

## How to run on Windows

After many trials, on Windows we recommend to install audiblez in a Python venv:

1. Open a Windows terminal
2. Create anew folder: `mkdir audiblez`
3. Enter the folder: `cd audiblez`
4. Create a venv: `python -m venv venv`
5. Activate the venv: `.\venv\Scripts\Activate.ps1`
6. Install the dependencies: `pip install audiblez pillow wxpython`
7. Now you can run `audiblez` or `audiblez-ui`
8. For Cuda support, you need to install Pytorch accordingly: https://pytorch.org/get-started/locally/


## Speed

By default the audio is generated using a normal speed, but you can make it up to twice slower or faster by specifying a speed argument between 0.5 to 2.0:

```
audiblez book.epub -v af_sky -s 1.5
```

## Supported Voices

Use `-v` option to specify the voice to use. Available voices are listed here.
The first letter is the language code and the second is the gender of the speaker e.g. `im_nicola` is an italian male voice.

[For hearing samples of Kokoro-82M voices, go here](https://claudio.uk/posts/audiblez-v4.html)

| Language                  | Voices                                                                                                                                                                                                                                     |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🇺🇸 American English     | `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| 🇬🇧 British English      | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`                                                                                                                                          |
| 🇪🇸 Spanish              | `ef_dora`, `em_alex`, `em_santa`                                                                                                                                                                                                           |
| 🇫🇷 French               | `ff_siwis`                                                                                                                                                                                                                                 |
| 🇮🇳 Hindi                | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`                                                                                                                                                                                                |
| 🇮🇹 Italian              | `if_sara`, `im_nicola`                                                                                                                                                                                                                     |
| 🇯🇵 Japanese             | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`                                                                                                                                                                         |
| 🇧🇷 Brazilian Portuguese | `pf_dora`, `pm_alex`, `pm_santa`                                                                                                                                                                                                           |
| 🇨🇳 Mandarin Chinese     | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`                                                                                                                                 |

For more detaila about voice quality, check this document: [Kokoro-82M voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)

## How to run on GPU

By default, audiblez runs on CPU. If you pass the option `--cuda` it will try to use the Cuda device via Torch.

Check out this example: [Audiblez running on a Google Colab Notebook with Cuda ](https://colab.research.google.com/drive/164PQLowogprWQpRjKk33e-8IORAvqXKI?usp=sharing]).

We don't currently support Apple Silicon, as there is not yet a Kokoro implementation in MLX. As soon as it will be available, we will support it.

## Manually pick chapters to convert

Sometimes you want to manually select which chapters/sections in the e-book to read out loud.
To do so, you can use `--pick` to interactively choose the chapters to convert (without running the GUI).


## Help page

For all the options available, check the help page `audiblez --help` and `audiblez-voice --help`.

## Tests

```bash
pip install pytest
pytest test/test_text.py test/test_normalize_ru.py test/test_audio_export.py test/test_pipeline.py \
    test/test_sources.py test/test_stress_ru.py test/test_parallel.py test/test_bench.py
```

## Author

by [Claudio Santini](https://claudio.uk) in 2025, distributed under MIT licence.

Related Article: [Audiblez v4: Generate Audiobooks from E-books](https://claudio.uk/posts/audiblez-v4.html)
