# Hướng dẫn làm truyện audio tiếng Nga cho YouTube với audiblez

Tài liệu này mô tả quy trình đầy đủ: từ file EPUB tiếng Nga đến các video ~2 tiếng/tập sẵn sàng đăng YouTube.

## 1. Cài đặt (một lần)

Khuyến nghị: máy có **GPU NVIDIA** (từ 8 GB VRAM, ưu tiên dòng RTX 30xx/40xx trở lên vì hỗ trợ bfloat16).
Trên CPU vẫn chạy được nhưng rất chậm: đo thực tế trên CPU Xeon 4 nhân mất khoảng **8 giờ máy cho 1 giờ audio**
(tức ~17 giờ cho một tập 2 tiếng). Không có GPU thì có thể thuê GPU theo giờ (Colab, RunPod, Vast.ai…).

```bash
sudo apt install ffmpeg sox            # Ubuntu/Debian. Trên Mac: brew install ffmpeg sox
python3.12 -m venv venv && source venv/bin/activate
# PyTorch có CUDA: xem lệnh đúng cho máy tại https://pytorch.org/get-started/locally/
pip install -e ".[asr]"                # [asr] = Whisper tự kiểm tra từng câu (rất nên cài)
pip install -U flash-attn --no-build-isolation   # tuỳ chọn, giúp GPU nhanh và nhẹ hơn
```

Lần chạy đầu tiên sẽ tự tải model Qwen3-TTS (~4,5 GB) và Whisper.

## 2. Chọn giọng đọc

Giọng là yếu tố quyết định của truyện audio, nên hãy dành thời gian cho bước này.

**Cách nhanh**: dùng giọng có sẵn `ru_male_narrator` (mặc định cho sách tiếng Nga) hoặc `ru_female_narrator`.

**Cách tốt nhất**: tự thiết kế vài giọng rồi nghe và chọn:

```bash
audiblez-voice design --preset ru_male_narrator -n 6
audiblez-voice design --preset ru_female_narrator -n 6
audiblez-voice design --preset ru_male_storyteller -n 6      # giọng ông kể chuyện, khàn ấm
audiblez-voice design --name tram_am --language ru \
    --instruct "A deep male bass voice, 50 years old, native Russian speaker, slow and dramatic, mysterious tone"
```

Mỗi ứng viên là một thư mục `voices/<tên>/candidate_XX/` chứa `reference.flac` (nghe thử file này).
Công cụ in ra cho mỗi ứng viên:

- **pitch** (cao độ trung bình): giọng nam trầm ~90–120 Hz, nam thường 120–155 Hz, nữ 165–255 Hz.
  Model thiết kế giọng *không phải lúc nào cũng làm đúng mô tả* (vd. yêu cầu nam trầm nhưng ra giọng cao),
  ứng viên lệch sẽ được đánh dấu "probably discard".
- **ASR error**: Whisper nghe lại có đúng chữ không (nên ~0–3%).

Nghe thử một đoạn văn dài hơn bằng giọng đã chọn:

```bash
audiblez-voice preview --voice-profile voices/ru_male_narrator/candidate_03 -o thu.wav
```

**Dùng giọng thật** (giọng của anh, hoặc giọng người đọc đã đồng ý/cấp phép): ghi âm 6–15 giây, phòng yên tĩnh,
không nhạc nền, đọc một đoạn có cả lời kể lẫn hội thoại, rồi:

```bash
audiblez-voice create --audio ghi_am.wav --text "Chính xác từng chữ đã đọc trong file" -o voices/toi
```

Lưu ý: không dùng `--speaker Ryan/Vivian...` cho tiếng Nga (các giọng có sẵn của Qwen đọc tiếng Nga bị giọng nước ngoài),
và không nên đọc cả cuốn bằng `--instruct` (thiết kế giọng mỗi câu một lần → giọng thay đổi giữa các câu).
Clone từ một giọng tham chiếu giữ **cùng một giọng suốt nhiều giờ**.

## 3. Kiểm tra văn bản trước khi tốn GPU

```bash
audiblez truyen.epub --dry-run
```

Lệnh này in bảng chương và tạo `truyen.audiblez/script.txt` — chính xác những gì sẽ được đọc (số đã đổi thành chữ,
viết tắt đã mở rộng, đã thêm chữ ё…). Đọc lướt để phát hiện:

- **Tên riêng / từ đọc sai**: tạo file `tu_dien.txt`, mỗi dòng `từ gốc = cách đọc`, ví dụ:
  ```
  Германн = Герман
  ИИ = искусственный интеллект
  re:\bгл\.\s?(\d+) = глава \1
  ```
  rồi thêm `--lexicon tu_dien.txt` vào mọi lệnh.
- **Chương thừa/thiếu**: dùng `--chapters 1-20` hoặc `-p` (chọn tay).

## 4. Nghe thử một chương

```bash
audiblez truyen.epub --voice-profile voices/ru_male_narrator/candidate_03 --chapters 1
```

Nếu muốn nhịp chậm rãi hơn: `--pause-scale 1.2` (ngắt nghỉ dài hơn) hoặc `--speed 0.95`.
Nếu thấy giọng quá "kịch" hoặc thỉnh thoảng lạc: `--temperature 0.7` (ổn định hơn, bớt biểu cảm).

## 5. Làm cả cuốn thành các tập ~2 tiếng cho YouTube

```bash
audiblez truyen.epub --voice-profile voices/ru_male_narrator/candidate_03 \
    --youtube --episode-hours 2 --batch-size 4 --lexicon tu_dien.txt
```

Kết quả trong thư mục hiện tại:

| File | Dùng để |
|---|---|
| `truyen_part01.mp4`, `truyen_part02.mp4`… | Video 1080p (ảnh bìa trên nền mờ) để upload YouTube |
| `truyen_part01_youtube.txt`… | Mốc thời gian các chương — dán vào phần mô tả video, YouTube tự tạo chapter |
| `truyen_part01.m4b`… | Bản audiobook có chương (nghe trên điện thoại, kiểm tra trước khi đăng) |
| `truyen_parts.txt` | Danh sách các tập, thời lượng, chương đầu/cuối |
| `truyen.audiblez/qa_report.txt` | Những câu không qua kiểm tra tự động — nên nghe lại |

Các tập được chia **đều nhau** (vd. 7 tiếng → 4 tập ~1h45) và luôn cắt ở ranh giới chương
(chương quá dài thì cắt ở ranh giới đoạn văn). Ảnh video lấy từ bìa sách, hoặc tự chọn bằng `--cover anh.jpg`.

## 6. Kiểm soát chất lượng

- Mỗi câu được kiểm tra tự động: tốc độ đọc bất thường (nuốt chữ / ảo giác), quá giới hạn độ dài, câm, rè,
  và Whisper nghe lại so với văn bản. Câu lỗi được **tạo lại với seed khác** (mặc định tối đa 3 lần).
- Câu vẫn lỗi sau 3 lần: được ghi vào `qa_report.txt` (kèm đường dẫn file). Nghe thử; nếu tệ, chạy lại với
  `--retry-failed` (chỉ tạo lại những câu đó, phần còn lại lấy từ cache).
- Âm lượng được chuẩn hoá -16 LUFS (chuẩn nội dung lời nói). Muốn to hơn kiểu YouTube nhạc: `--loudness -14`.
- Bị tắt máy giữa chừng? Chạy lại đúng lệnh cũ: mọi câu đã tạo nằm trong cache `truyen.audiblez/cache/`.
- Sửa một đoạn trong EPUB hoặc thêm từ vào lexicon: chỉ những câu thay đổi được tạo lại.

## 7. Mẹo cho kênh YouTube

- Giữ **cùng một giọng** (cùng thư mục voice profile) cho cả bộ truyện để khán giả quen giọng.
- Lưu lại thư mục `voices/...` và file `tu_dien.txt` — đó là "tài sản" của kênh.
- Qwen3-TTS và Kokoro đều dùng giấy phép Apache-2.0 (được phép dùng thương mại). Bản quyền **nội dung truyện** là
  trách nhiệm của người đăng: dùng tác phẩm đã hết bản quyền hoặc đã được phép.
- Nếu clone giọng người thật, phải có sự đồng ý của người đó.
