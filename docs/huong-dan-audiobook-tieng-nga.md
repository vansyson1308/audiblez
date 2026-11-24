# Hướng dẫn làm truyện audio tiếng Nga cho YouTube với audiblez

Tài liệu này mô tả quy trình đầy đủ: từ kịch bản truyện (`.txt`, `.md`, `.docx`) hoặc sách (`.epub`, `.fb2`) tiếng Nga
đến video ~2 tiếng sẵn sàng đăng YouTube, với kiểu nội dung "жизненные истории" (truyện đời thường: tình yêu,
phản bội, gia đình…) dài 1–2,5 giờ mỗi video.

## 0. Tóm tắt: cách nào nhanh nhất?

| Máy | Lệnh | Tốc độ (ước tính) | Một video 2 giờ |
|---|---|---|---|
| CPU 4 nhân (máy hiện tại) | `audiblez truyen.txt --youtube` | ~0,12x thời gian thực (đo thực tế) | ~16 giờ |
| **Kaggle miễn phí, GPU T4 ×2** | notebook `kaggle/audiblez_kaggle.ipynb` | ~0,9x mỗi T4 × 2 GPU (số của tác giả faster-qwen3-tts) | **~1,1 giờ** |
| RTX 4090 thuê (~$0,3/giờ) | `--backend faster` | ~4,2x (số của tác giả) | ~30 phút |

Tốc độ thực phụ thuộc GPU, độ chính xác số (dtype) và kích thước batch: **hãy đo bằng `--benchmark` trước khi tin
số liệu** (mục 6). Mọi bước đều tiếp tục được nếu bị ngắt: chạy lại đúng lệnh, phần đã xong lấy từ cache.

## 1. Cài đặt (một lần)

```bash
sudo apt install ffmpeg            # Ubuntu/Debian. Trên Mac: brew install ffmpeg
python3.12 -m venv venv && source venv/bin/activate
# PyTorch có CUDA: xem lệnh đúng cho máy tại https://pytorch.org/get-started/locally/
pip install "audiblez[qwen,asr,stress] @ git+https://github.com/vansyson1308/audiblez.git"
```

`qwen` = Qwen3-TTS bản gốc (CPU, mọi GPU); `asr` = Whisper tự kiểm tra từng câu; `stress` = RUAccent (trọng âm).

Python 3.10–3.13 đều được (Kaggle và Colab hiện dùng Python 3.13; Kokoro — chỉ dùng cho tiếng Anh/Ý/… — chưa hỗ trợ
3.13 nên không được cài ở đó, tiếng Nga không cần Kokoro). Lần chạy đầu sẽ tự tải model Qwen3-TTS (~4,5 GB) và Whisper.

**Backend "faster" (chỉ trên GPU NVIDIA)**: [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
dùng CUDA graphs, nhanh gấp ~2 lần trên T4 và ~5 lần trên RTX 4090 cho mỗi GPU (số của tác giả). Cài `fast` **thay
cho** `qwen` (hai gói cùng cung cấp module `qwen_tts`, không cài chung một môi trường):

```bash
pip install "audiblez[fast,asr,stress] @ git+https://github.com/vansyson1308/audiblez.git"
```

Khi có GPU, audiblez tự dùng backend này và **tự chuyển về backend gốc nếu nó lỗi** (`--backend qwen-tts` để tắt
hẳn). Extra `fast` ghim transformers 5.15–5.16: `qwen-tts-hf` (bản Qwen3-TTS mà faster-qwen3-tts dùng) không nạp
được model với transformers ≥ 5.17. Hai backend dùng **chung cache**: câu tạo trên Kaggle có thể tạo lại (retake)
trên máy nhà bằng backend gốc.

## 2. Chuẩn bị kịch bản truyện

audiblez đọc trực tiếp:

- **`.txt`**: mỗi dòng là một đoạn, hoặc văn bản ngắt dòng cứng kiểu PDF/Gutenberg (tự ghép lại). Dòng `Глава 1`,
  `I`, `Пролог`… → chương (mốc thời gian YouTube). Dòng `* * *` → chuyển cảnh (ngắt nghỉ dài). Thoại gõ bằng gạch nối
  `- Привет, - сказала она.` được đọc như gạch ngang `—`. Mã hoá UTF-8 hoặc Windows-1251 đều được.
- **`.md`**: `# Tiêu đề`, `## Глава 1`, `---` = chuyển cảnh; `**đậm**`, `*nghiêng*`, link được làm sạch.
- **Tiêu đề, tác giả, ảnh bìa** của truyện (đọc ở đầu video, ghi vào file m4b): khối front matter ở đầu file
  ```
  ---
  title: Брак на спор
  author: По ту сторону правды
  ---
  ```
  hoặc dòng `# Tiêu đề` duy nhất của file `.md`; không có thì lấy tên file. Ảnh `cover.jpg` cạnh file truyện được dùng
  làm ảnh video.
- **Một thư mục** chứa nhiều `.txt`/`.md`: mỗi file là một chương (theo thứ tự tự nhiên: 2 trước 10).
- **`.docx`** (Word, Google Docs → Tải xuống → .docx): style *Title* = tiêu đề, *Heading 1/2* = chương.
- **`.fb2`, `.fb2.zip`** (định dạng ebook phổ biến nhất ở Nga) và **`.epub`**: tiêu đề, tác giả, bìa, chương, bỏ chú thích.

Tiêu đề viết HOA kiểu YouTube (`ОН НЕ ОЖИДАЛ, ЧТО ПОЛЮБИТ СВОЮ ЖЕНУ`) được tự chuyển về chữ thường khi đọc (TTS có thể
đánh vần chữ in hoa như chữ viết tắt); viết tắt thật (МВД, СССР, США) được giữ nguyên.

## 3. Kiểm tra văn bản trước khi tốn GPU

```bash
audiblez truyen.txt --dry-run
audiblez truyen.txt --dry-run --stress ruaccent      # xem luôn trọng âm sẽ đọc
```

Lệnh này in bảng chương và tạo `truyen.audiblez/script.txt` — chính xác những gì sẽ được đọc (số đã đổi thành chữ
đúng cách/giống, viết tắt đã mở rộng, đã thêm chữ ё, dấu trọng âm…). Đọc lướt để phát hiện:

- **Tên riêng / từ đọc sai**: tạo file `tu_dien.txt`, mỗi dòng `từ gốc = cách đọc`, ví dụ:
  ```
  Германн = Герман
  ИИ = искусственный интеллект
  замок = замо́к
  re:\bгл\.\s?(\d+) = глава \1
  ```
  rồi thêm `--lexicon tu_dien.txt` vào mọi lệnh. Với `--stress ruaccent`, dấu trọng âm trong từ điển (`замо́к`) được
  giữ nguyên và model đọc theo.
- **Chương thừa/thiếu**: dùng `--chapters 1-20` hoặc `-p` (chọn tay).

## 4. Trọng âm (khuyến nghị thử): `--stress ruaccent`

Lỗi dễ nghe thấy nhất của TTS tiếng Nga là **đặt sai trọng âm** (за́мок/замо́к, сто́ит/стои́т, все/всё, tên riêng). Với
`--stress ruaccent`:

1. [RUAccent](https://github.com/Den4ikAI/ruaccent) (MIT) đặt dấu trọng âm cho từng từ theo ngữ cảnh và khôi phục ё
   (chạy trên CPU, ~10–100 ms mỗi câu, có cache; lần đầu tải ~1,5 GB model).
2. Văn bản được đọc bằng [siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf),
   bản fine-tune Apache-2.0 của chính Qwen3-TTS 1.7B được huấn luyện để **làm theo dấu trọng âm** (theo model card: 96–100%
   so với 17–65% của model gốc). Cùng giọng clone, cùng tốc độ.

Thử trên bộ 24 câu có từ đồng hình (за́мок/замо́к, му́ки/муки́, бе́лки/белки́, доро́гой/дорого́й…), RUAccent đặt đúng cả 24
khi mỗi câu có một từ đồng hình. Lưu ý:

- RUAccent vẫn sai khoảng 2–3% số từ (số của tác giả model): sửa bằng `--lexicon` khi nghe thấy.
- **Model card cảnh báo**: một phần câu dùng để huấn luyện lấy từ bộ dữ liệu `biggest-ru-book` không khai báo giấy
  phép; tác giả khuyên kiểm tra trước khi dùng thương mại. Trọng số model là Apache-2.0.
- Hãy nghe so sánh một chương có và không có `--stress ruaccent` trước khi làm cả loạt.

## 5. Chọn giọng đọc

Giọng là "tài sản" của kênh: dùng **cùng một giọng** cho mọi video.

**Cách nhanh**: giọng có sẵn `ru_male_narrator` (mặc định) hoặc `ru_female_narrator` (`--voice-profile ru_female_narrator`).
Truyện đời thường kể từ góc nhìn nhân vật nữ thường hợp giọng nữ ấm.

**Cách tốt nhất**: tự thiết kế vài giọng rồi nghe và chọn:

```bash
audiblez-voice design --preset ru_female_narrator -n 6
audiblez-voice design --preset ru_male_storyteller -n 6      # giọng ông kể chuyện, khàn ấm
audiblez-voice design --name tram_am --language ru \
    --instruct "A deep male bass voice, 50 years old, native Russian speaker, slow and dramatic, mysterious tone"
audiblez-voice preview --voice-profile voices/ru_female_narrator/candidate_03 -o thu.wav
```

Mỗi ứng viên in ra **pitch** (nam trầm ~90–120 Hz, nam 120–155 Hz, nữ 165–255 Hz; ứng viên lệch được đánh dấu
"probably discard") và **ASR error** (nên ~0–3%).

**Dùng giọng thật** (giọng của anh, hoặc giọng người đọc đã đồng ý/cấp phép): ghi âm 6–15 giây, phòng yên tĩnh,
không nhạc nền, đọc một đoạn có cả lời kể lẫn hội thoại, rồi:

```bash
audiblez-voice create --audio ghi_am.wav --text "Chính xác từng chữ đã đọc trong file" -o voices/toi
```

Không dùng `--speaker Ryan/Vivian…` cho tiếng Nga (giọng nước ngoài), và không đọc cả truyện bằng `--instruct` (giọng
thay đổi giữa các câu). Clone từ một giọng tham chiếu giữ **cùng một giọng suốt nhiều giờ**.

## 6. Đo cấu hình nhanh nhất trên GPU của mình

```bash
audiblez truyen.txt --voice-profile ru_female_narrator --benchmark            # thêm --stress ruaccent nếu dùng
```

Công cụ lấy vài câu của chính truyện, đo từng độ chính xác số (float32 / bfloat16 / float16), backend (faster / gốc) và
kích thước batch trên GPU đang có; **mọi câu đều qua cùng bộ kiểm tra chất lượng** như khi render thật (tốc độ đọc,
im lặng, NaN, độ giống giọng, Whisper), nên cấu hình làm hỏng âm thanh không bao giờ được khuyến nghị. Nó in dòng
`Fastest: --backend … --batch-size … --dtype …` để dùng cho lệnh render. Ở nửa độ chính xác (bf16/fp16), audiblez giữ
bộ giải mã âm thanh (codec) ở float32 để tránh câu bị im lặng/nhiễu.

## 7. Render

Một chương để nghe thử:

```bash
audiblez truyen.txt --voice-profile ru_female_narrator --chapters 1
```

Cả truyện thành video YouTube:

```bash
audiblez truyen.txt --voice-profile ru_female_narrator --youtube --lexicon tu_dien.txt --stress ruaccent
```

- **Nhiều GPU**: khi có từ 2 GPU, audiblez tự chạy **mỗi GPU một tiến trình** lấy việc từ một hàng đợi chung (GPU
  nào rảnh thì nhận câu tiếp), vd. 2 T4 của Kaggle → nhanh gấp đôi. `--gpus off` để tắt, `--gpus 0,1` để chọn.
- **Giới hạn thời gian**: `--time-limit 11.5` dừng gọn sau 11,5 giờ (phiên Kaggle tối đa 12 giờ); chạy lại cùng lệnh
  để làm tiếp.
- Truyện rất dài: `--episode-hours 2` chia thành các tập ~2 giờ (cắt ở ranh giới chương/đoạn).
- Nhịp chậm rãi hơn: `--pause-scale 1.2` hoặc `--speed 0.95`. Giọng quá "kịch" hoặc thỉnh thoảng lạc: `--temperature 0.7`.

Kết quả:

| File | Dùng để |
|---|---|
| `truyen.mp4` (hoặc `truyen_part01.mp4`…) | Video 1080p (ảnh bìa trên nền mờ) để upload YouTube; `--cover anh.jpg` để chọn ảnh |
| `truyen_youtube.txt` | Mốc thời gian các chương — dán vào mô tả video (YouTube cần ≥ 3 mốc, mốc đầu 00:00) |
| `truyen.m4b` | Bản audiobook có chương (nghe kiểm tra trên điện thoại) |
| `truyen.audiblez/script.txt` | Văn bản thực sự được đọc |
| `truyen.audiblez/qa_report.txt` | Những câu không qua kiểm tra tự động (kể cả từ các lần chạy trước) — nên nghe lại |

## 8. Chạy miễn phí trên Kaggle (2 × T4)

Xem `kaggle/README.md`: tạo dataset chứa các file truyện (+ giọng, từ điển, bìa), mở notebook
`kaggle/audiblez_kaggle.ipynb`, chạy nền ("Save & Run All"), tải kết quả. Notebook render **một hoặc nhiều truyện**
(`BOOK = [...]`) trên cả hai GPU, tự dừng trước khi hết phiên 12 giờ, và có thể nối tiếp từ output của lần chạy trước.
Quota ~30 giờ/tuần: với ~1,1 giờ cho một video 2 giờ là đủ cho khoảng 1 video/ngày (cần đo thực tế bằng benchmark).

## 9. Kiểm soát chất lượng

- Mỗi câu được kiểm tra tự động: tốc độ đọc bất thường (nuốt chữ / ảo giác), quá giới hạn độ dài, câm, rè, cụt
  đuôi câu, giọng khác người đọc (so với giọng tham chiếu), và Whisper nghe lại so với văn bản. Câu lỗi được
  **tạo lại với seed khác** (mặc định tối đa 3 lần).
- Câu vẫn lỗi sau 3 lần: được ghi vào `qa_report.txt` (kèm đường dẫn file) — kể cả ở các lần chạy sau lấy từ cache,
  cho đến khi một lần tạo lại đạt. Nghe thử; nếu tệ, chạy lại với `--retry-failed` (chỉ tạo lại những câu đó, phần
  còn lại lấy từ cache) — có thể làm trên máy nhà (CPU), mỗi câu ~2–3 phút.
- Âm lượng được chuẩn hoá -16 LUFS (chuẩn nội dung lời nói). Muốn to hơn: `--loudness -14`.
- **Room tone**: khoảng ngắt giữa các câu được lấp bằng chính tiếng nền rất nhỏ của giọng đọc (đo từ audio, khoảng
  -65 dBFS) thay vì im lặng tuyệt đối, như bản ghi âm thật: trên tai nghe, nền âm không còn "tắt-bật" sau mỗi câu
  (chuẩn ACX của audiobook cũng yêu cầu vậy). Tắt bằng `--no-room-tone`.
- Bị tắt máy giữa chừng? Chạy lại đúng lệnh cũ: mọi câu đã tạo nằm trong cache `truyen.audiblez/cache/` (ghi nguyên
  tử: không bao giờ có file hỏng dở dang).
- Sửa một đoạn hoặc thêm từ vào lexicon: chỉ những câu thay đổi được tạo lại.

## 10. Lưu ý cho kênh YouTube

- Giữ **cùng một giọng** (cùng thư mục voice profile) và cùng `tu_dien.txt` cho mọi video.
- Qwen3-TTS (và bản fine-tune trọng âm) dùng giấy phép Apache-2.0. Bản quyền **nội dung truyện** là trách nhiệm của
  người đăng: dùng truyện tự viết, đã được phép, hoặc tác phẩm đã hết bản quyền (ở Nga **và** ở Mỹ).
- Chính sách kiếm tiền của YouTube xếp "chỉ đọc lại tài liệu không do bạn sáng tạo" và nội dung "theo khuôn, sản xuất
  hàng loạt" vào loại không được kiếm tiền: nên có lời dẫn/bình luận riêng, hình minh hoạ, và không đăng hàng loạt
  video giống hệt nhau.
- Nếu clone giọng người thật, phải có sự đồng ý của người đó; giọng clone của chính mình thì YouTube không bắt buộc
  công bố "nội dung tổng hợp".
