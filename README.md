# Audiblez: Generate  audiobooks from e-books

[![Installing via pip and running](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml)
[![Git clone and run](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/audiblez)
![PyPI - Version](https://img.shields.io/pypi/v/audiblez)

### v6: your own stories in Japanese, Korean, Russian… rendered fast on free GPUs

Audiblez turns `.epub` and `.fb2` e-books **and your own stories** (`.txt`, `.md`, `.docx`, or a folder with one file
per chapter) into
`.m4b` audiobooks and `.mp4` videos for YouTube, with a natural, expressive narrator.

| Engine | Languages | Strengths |
|---|---|---|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B (Apache-2.0) | 🇯🇵 🇰🇷 🇷🇺 🇬🇧 🇩🇪 🇫🇷 🇪🇸 🇮🇹 🇵🇹 🇨🇳 | very natural and expressive, one cloned narrator for hours |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | 🇺🇸 🇬🇧 🇪🇸 🇫🇷 🇮🇳 🇮🇹 🇯🇵 🇧🇷 🇨🇳 | very fast, runs well on CPU |

Japanese, Korean and Russian use Qwen3-TTS with a bundled native narrator voice (`ja_male_narrator`,
`ja_female_narrator`, `ko_male_narrator`, `ko_female_narrator`, `ru_male_narrator`, `ru_female_narrator`); other
languages use Kokoro when it supports them. Override with `--engine`, `--voice-profile`, `-v`.

What happens between the text and the audio file matters as much as the voice:

- **Clean text**: EPUBs in reading order without footnote markers, notes, tables of contents or copyright pages;
  books stored as a single file are split into their real chapters. Your own `.txt`/`.md` files are read with their
  chapter headings (`# Title`, `第1話`, `第一章`, `제1장`, `1화`, `Глава 5`, `Chapter 3`…), scene breaks (`◇◇◇`,
  `* * *`, `---`) and an optional front matter (`title:`, `author:`, `language:`, `cover:`); a short first line
  followed by a blank line is the title of a `.txt` story (a file name is never read aloud). Shift_JIS, EUC-KR, GB18030
  and cp1251 files are detected automatically.
- **Japanese**: furigana written like on 小説家になろう / カクヨム / 青空文庫 (`｜颯《はやて》`, `魔王城《まおうじょう》`,
  `東京（とうきょう）`) and EPUB `<ruby>` are **read with their reading** — the reliable way to get names and rare kanji
  right; width normalization (`２０２６` → `2026`, half-width katakana), `【】`, `〜`, `――`, emphasis marks, sentence
  splitting that keeps `「…」と言った` together, chunks cut at `、` / `」`, never inside a word.
- **Korean**: numbers spelled in Hangul with the right system for the counter (`3명` → *세 명*, `2026년 6월` →
  *이천이십육 년 유월*, `오후 3시 30분` → *오후 세 시 삼십 분*, `20살` → *스무 살*), sentence splitting on Korean
  punctuation, spacing kept when lines are joined.
- **Russian**: numbers in the right case and gender, dates, Roman numerals, abbreviations, restoration of `ё`,
  ALL-CAPS YouTube-style titles read as words (abbreviations such as МВД kept), dialogue typed with hyphens.
  `--stress ruaccent` marks the stressed vowel of every word from its context with
  [RUAccent](https://github.com/Den4ikAI/ruaccent) (за́мок / замо́к, все / всё) and reads it with
  [a Qwen3-TTS fine-tune that follows the marks](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf)
  (Apache-2.0; its model card warns that part of its training texts has no declared license).
  FB2 e-books (`.fb2`, `.fb2.zip`) are read with their chapters, epigraphs and cover, without notes.
- **Pronunciation dictionary** (`--lexicon`): lines like `颯 = はやて`, `魔王城 = まおうじょう`, `Германн = Герман`
  (works inside Japanese text, which has no spaces). `--dry-run` writes the exact normalized script to check first.
- **Natural pacing**: sentences are synthesized in chunks of about 15-20 s within a paragraph (never across), the
  model's silences are trimmed and controlled pauses inserted between sentences, paragraphs, headings, scene breaks
  and chapters; loud or quiet sentences are gently leveled. `--pause-scale` makes the reading calmer or faster.
  The pauses carry the voice's own soft background ("room tone", measured per chapter) instead of digital silence,
  so the background does not switch off between sentences (`--no-room-tone` to disable).
- **Automatic quality control**: every chunk is checked (speaking rate for its language, runaway generation, silence,
  clipping, **clipped sentence endings**, **similarity with the narrator's voice**, and a Whisper transcription
  compared with the text — through kana readings for Japanese). Failed chunks are regenerated with another seed; the
  remaining ones are listed in `qa_report.txt` (across runs and parallel workers, until a retake passes;
  `--retry-failed` regenerates them).
- **Fast**: one worker per GPU (both T4 GPUs of a free Kaggle notebook), the CUDA-graph backend
  [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) with automatic fallback, batching, and
  `--benchmark` to measure the fastest precision, backend and batch size on your GPU (every take is quality
  checked, so a faster setting that damages the audio is never recommended) instead of guessing.
- **Resumable**: every chunk is cached; an interrupted run (or a 12-hour Kaggle session) continues where it stopped,
  and editing one paragraph only re-synthesizes that paragraph. `--time-limit 11.5` stops cleanly before a cloud
  session ends. With half precision (bf16/fp16) the speech codec stays in float32 (no silent or noisy takes).
- **Mastering and YouTube**: -16 LUFS / -1.5 dBTP with a linear gain and peak limiter, `.m4b` with chapters and cover,
  `--youtube` for a 1080p `.mp4` and the chapter timestamps of the description, `--episode-hours 2` for ~2-hour parts.

![Audiblez GUI on MacOSX](./imgs/mac.png)

## Quick start: a Japanese or Korean story

```bash
sudo apt install ffmpeg                                    # (brew install ffmpeg on Mac)
pip install "audiblez[qwen,asr,ja] @ git+https://github.com/vansyson1308/audiblez.git"   # CPU / any GPU
# or, on an NVIDIA GPU, the CUDA-graph backend instead of [qwen]:
pip install "audiblez[fast,asr,ja] @ git+https://github.com/vansyson1308/audiblez.git"

audiblez story.md --dry-run                                # 1. check chapters and story.audiblez/script.txt
audiblez story.md --chapters 1                             # 2. listen to the first chapter
audiblez story.md --youtube --episode-hours 2              # 3. ~2 h YouTube videos + .m4b audiobooks
audiblez story.md --voice-profile ko_female_narrator       #    another narrator (audiblez-voice list)
```

`qwen` and `fast` both provide the `qwen_tts` module: install one of them per environment, never both.

`story.md` can look like this (or be a plain `.txt` with one paragraph per line, or a folder of chapter files):

```markdown
---
title: 星を継ぐ者
author: 山田太郎
---
# 第1話　出会い

　雨が降っていた。｜颯《はやて》は傘を持っていなかった。
「待って！」と彼女は叫んだ。

◇◇◇

# 第2話　約束
```

## Fast rendering on Kaggle (free 2× T4)

Qwen3-TTS 1.7B needs a GPU for long books (on a 4-core CPU, 1 hour of audio takes about 8 hours). Kaggle gives about
30 GPU-hours per week with two T4 GPUs. Open [`kaggle/audiblez_kaggle.ipynb`](kaggle/audiblez_kaggle.ipynb) on
Kaggle (or push it with `kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4` after editing
`kaggle/kernel-metadata.json`), select *GPU T4 x2* and *Internet on*, add your story as a dataset, set `BOOK` and
`VOICE`, and run *Save & Run All*. The notebook installs audiblez, checks the text, benchmarks precision (float32 / bfloat16 /
float16), backends and batch sizes on the T4s, keeping only settings whose takes pass the quality checks, renders
with one worker per GPU and saves the videos, audiobooks and chapter lists in the *Output* tab. A run that
hits the 12-hour limit can be resumed from its output (`RESUME_FROM`).

Step-by-step guides (Vietnamese): [Japanese / Korean stories on Kaggle](docs/huong-dan-audiobook-nhat-han.md),
[Russian audiobooks](docs/huong-dan-audiobook-tieng-nga.md).

## Voices

`audiblez-voice list` shows the bundled voices and presets. For the best result, design (or record) your own
narrator and use it for the whole series:

```bash
audiblez-voice design --preset ja_male_narrator -n 6           # 6 candidates in ./voices/ja_male_narrator/
audiblez-voice design --name grandpa --language ko \
    --instruct "An older male voice, 65 years old, native Korean, husky, warm, unhurried storyteller"
audiblez-voice create --audio my_voice.wav --text "Exact transcript of the recording" -o voices/me --language ja
audiblez-voice preview --voice-profile voices/ja_male_narrator/candidate_03 --language ja -o test.wav
audiblez story.md --voice-profile voices/ja_male_narrator/candidate_03 --youtube --episode-hours 2
```

Each candidate prints its length, pitch (male voices 85-155 Hz, a deep baritone 90-120 Hz, female 165-255 Hz),
speaking rate and Whisper error: the voice design model does not always follow the description, so listen and keep
the one you like. A voice profile is a folder with a 6-15 s clean reference (`reference.flac`) and its exact
transcript (`reference.txt`). Cloning keeps the same native voice for hours; the built-in Qwen3-TTS speakers
(`--speaker Ono_Anna`, `--speaker Sohee`…) are an alternative with style instructions (`--instruct`). Only clone
voices you have the rights to.

Useful options for long books: `--lexicon names.txt`, `--temperature 0.7` (steadier, less expressive),
`--retry-failed` (regenerate the chunks of the QA report), `--loudness -14` (louder masters), `--cover image.jpg`,
`--gpus 0,1` / `--gpus off`, `--backend faster|qwen-tts`, `--batch-size 8` (large GPUs).

## How to install the Command Line tool

If you have Python 3.10-3.13 on your computer, you can install it with pip (Kokoro is only installed on Python
3.10-3.12; on 3.13, as on Kaggle and Colab, Qwen3-TTS is used).
You also need `espeak-ng` (Kokoro) and `ffmpeg` installed on your machine:

```bash
sudo apt install ffmpeg espeak-ng                   # on Ubuntu/Debian 🐧
pip install "audiblez[qwen,asr]"                    # [qwen]: Qwen3-TTS engine, [asr]: Whisper quality control
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
pytest test/test_text.py test/test_normalize_ru.py test/test_audio_export.py test/test_pipeline.py test/test_cjk.py \
    test/test_formats.py test/test_stress_ru.py test/test_reliability.py
```

## Author

by [Claudio Santini](https://claudio.uk) in 2025, distributed under MIT licence.

Related Article: [Audiblez v4: Generate Audiobooks from E-books](https://claudio.uk/posts/audiblez-v4.html)
