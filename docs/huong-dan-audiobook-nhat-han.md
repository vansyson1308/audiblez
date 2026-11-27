# Hướng dẫn làm truyện audio tiếng Nhật / tiếng Hàn cho YouTube (render miễn phí trên Kaggle)

Quy trình đầy đủ: **tự viết truyện** (file `.md` hoặc `.txt`) → kiểm tra văn bản → render trên **2 GPU T4 miễn phí
của Kaggle** → tải về video `.mp4` (~2 tiếng/tập) kèm danh sách chương để dán vào mô tả YouTube.

## 1. Viết truyện đúng định dạng

Một file Markdown cho cả truyện (hoặc một thư mục, mỗi chương một file `.md`/`.txt`, sắp theo thứ tự tự nhiên:
chương 2 đứng trước chương 10):

```markdown
---
title: 星降る駅
author: 山田太郎
---
# 第1話　最終列車

　その夜、駅のホームには誰もいなかった。｜颯《はやて》は時計を見た。午後11時30分、最終列車まであと5分。

「遅かったじゃない」と、背後から声がした。

◇◇◇

「どうして、ここに？」

# 第2話　約束

　二人は線路の向こうに広がる夜空を見上げた。
```

- **Phần đầu** (`---` … `---`, không bắt buộc): `title`, `author`, `language` (`ja`/`ko`; thường tự nhận ra), `cover`
  (ảnh bìa cho video, đường dẫn tính từ file truyện).
- **Mỗi tiêu đề `#` là một chương** (`# 第1話　タイトル`, `# 제1화 제목`, `# プロローグ`, `# 에필로그`…). Tên chương được
  đọc lên, có khoảng nghỉ trước/sau, và thành mốc chương trong file m4b và trong mô tả YouTube.
- **Đổi cảnh**: một dòng chỉ có `◇◇◇`, `* * *`, `***` hoặc `---` → khoảng lặng dài hơn.
- **Mỗi đoạn văn cách nhau một dòng trống** (kiểu web novel, mỗi dòng một đoạn, cũng được).
- File lưu UTF-8 là tốt nhất; Shift_JIS, EUC-JP, EUC-KR/CP949 cũng tự nhận ra.

### Tiếng Nhật: cách đọc chữ Hán (quan trọng nhất)

Model TTS đôi khi đọc sai chữ Hán, nhất là **tên riêng** và từ hiếm. Cách chắc chắn nhất là ghi cách đọc ngay trong
truyện, theo kiểu của 小説家になろう / カクヨム / 青空文庫:

| Viết trong truyện | Được đọc là |
|---|---|
| `｜颯《はやて》` (có dấu ｜ ở đầu) | はやて |
| `魔王城《まおうじょう》` (cả cụm chữ Hán đứng ngay trước 《》) | まおうじょう |
| `東京（とうきょう）` (ngoặc tròn toàn góc, chỉ chứa kana) | とうきょう |
| `《《大切》》` (nhấn mạnh), `［＃改ページ］` (chú thích) | 大切 / bị bỏ |

Chỉ cần ghi cách đọc **lần đầu** cho tên nhân vật nếu thêm tên đó vào **từ điển phát âm** (mục 3): từ điển áp dụng
cho cả truyện.

Số được tự đổi thành cách đọc đúng theo lượng từ: `3本` → さんぼん, `4人` → よにん, `14日` → じゅうよっか,
`7時30分` → しちじさんじゅっぷん, `第1話` → だいいちわ, `50%` → ごじゅっパーセント. Chữ số toàn góc (`３本`) và
katakana nửa góc cũng được xử lý.

### Tiếng Hàn

- Viết đúng chính tả và khoảng cách (띄어쓰기): model đọc theo đúng khoảng cách đó.
- Số được đổi đúng hệ đếm: thuần Hàn với 명/개/살/시/번… (`3명` → 세 명, `20살` → 스무 살, `10:30` → 열 시 삼십 분),
  Hán-Hàn với 년/월/일/분/원/층… (`2026년` → 이천이십육 년, `6월` → 유월), và `제3장` → 제삼장, `2개월` → 이 개월.
- Lỗi đã biết của Qwen3-TTS với tiếng Hàn là **mất âm tiết cuối câu**: mỗi câu được Whisper nghe lại, câu nào thiếu
  đuôi sẽ tự tạo lại.

## 2. Kiểm tra văn bản trước khi tốn GPU

Trên máy bất kỳ (không cần GPU), hoặc ô thứ 1 của notebook Kaggle:

```bash
audiblez truyen.md --dry-run
```

Lệnh in bảng chương và tạo `truyen.audiblez/script.txt`: **chính xác những gì sẽ được đọc** (số đã thành chữ, furigana
đã thay vào). Đọc lướt file này để tìm từ cần thêm vào từ điển.

## 3. Từ điển phát âm (`--lexicon`)

File `tu_dien.txt`, mỗi dòng `từ gốc = cách đọc`, áp dụng cho cả truyện:

```
颯 = はやて
魔王城 = まおうじょう
AI = エーアイ
```

Tiếng Hàn, ví dụ: `AI = 에이아이`, `SNS = 에스엔에스`. Với tiếng Nhật, mục từ được thay **ở mọi chỗ xuất hiện**, kể cả
bên trong từ khác (`颯 = はやて` cũng đổi 颯爽 thành はやて爽): khi chữ đó còn dùng trong từ khác, ghi cả cụm
(`颯は = はやては`) hoặc dùng furigana ngay trong truyện.

Dùng `--lexicon tu_dien.txt` (trên Kaggle: `EXTRA = '--lexicon /kaggle/input/<dataset>/tu_dien.txt'`).

## 4. Giọng đọc

Có sẵn 4 giọng bản ngữ (người đọc kể chuyện, clone ổn định suốt nhiều giờ):

| Giọng | Mô tả |
|---|---|
| `ja_male_narrator` | Nam, tiếng Nhật, trầm ấm, nhịp kể tự nhiên (mặc định cho tiếng Nhật) |
| `ja_female_narrator` | Nữ, tiếng Nhật, nhẹ nhàng, rõ chữ |
| `ko_male_narrator` | Nam, tiếng Hàn, trầm (mặc định cho tiếng Hàn) |
| `ko_female_narrator` | Nữ, tiếng Hàn, ấm |

Muốn giọng riêng cho kênh: tạo vài ứng viên, nghe `reference.flac` của từng ứng viên rồi chọn:

```bash
audiblez-voice design --preset ja_male_narrator -n 6
audiblez-voice design --name ojisan --language ja \
    --instruct "An older male voice, 60 years old, native Japanese, husky, warm, unhurried storyteller"
audiblez-voice preview --voice-profile voices/ojisan/candidate_03 --language ja -o thu.wav
```

**Dùng một giọng cho cả series** để khán giả quen giọng kênh. Chỉ clone giọng thật khi có sự đồng ý của người đó.

## 5. Render trên Kaggle (miễn phí, 2× T4)

**Chuẩn bị một lần:**

1. Tạo tài khoản kaggle.com và **xác minh số điện thoại** (bắt buộc để bật GPU và Internet).
2. *Create → New Dataset*: tải lên file truyện (và `tu_dien.txt`, ảnh bìa nếu có). Ví dụ đặt tên `my-story`.
3. *Create → New Notebook → File → Import Notebook*: chọn file `kaggle/audiblez_kaggle.ipynb` của repo
   (hoặc dán link GitHub của file).
4. Thanh bên phải: *Session options → Accelerator → GPU T4 x2*, *Internet → On*,
   *Add Input* → chọn dataset `my-story`.

**Mỗi lần render:**

1. Ô *Settings*: đặt `BOOK` (bấm biểu tượng copy đường dẫn của file trong mục *Input*), `VOICE`, `EPISODE_HOURS`.
2. *Save Version → Save & Run All (Commit)*. Notebook chạy nền, đóng trình duyệt cũng được.
3. Xong thì vào trang notebook → tab *Output*: `*_part01.mp4`, `*_part02.mp4`… (video ~2 tiếng),
   `*.m4b` (sách nói có mốc chương), `*_youtube.txt` (danh sách mốc thời gian chương để dán vào mô tả video).

Notebook tự làm: cài đặt → kiểm tra văn bản → **đo tốc độ** (độ chính xác số float32/bfloat16/float16, backend,
batch) trên chính GPU đó, chỉ giữ cấu hình mà các câu thử vẫn qua kiểm tra chất lượng (mất ~10–15 phút; lần sau đặt
`RUN_BENCHMARK = False` và chép các giá trị `Fastest` đã in ra vào `BACKEND, BATCH, DTYPE`) →
render với **một tiến trình cho mỗi GPU** (2 GPU chạy song song) → Whisper nghe lại từng câu, câu lỗi tự tạo lại →
master âm lượng chuẩn YouTube (-16 LUFS) → xuất video.

**Hạn mức và chạy tiếp:**

- Kaggle cho khoảng **30 giờ GPU/tuần**, mỗi phiên tối đa **12 giờ**. Dòng `Progress` trong log in tốc độ thật
  (`x real time (all workers)`) và thời gian còn lại (`ETA`).
- Phiên hết giờ khi chưa xong: không mất gì. Mở notebook, *Add Input* → output của lần chạy trước, đặt
  `RESUME_FROM = '/kaggle/input/<tên-notebook>/output'`, chạy lại: chỉ những câu còn thiếu được tạo.
- Truyện dài: render từng phần với `EXTRA = '--chapters 1-10'`, rồi `--chapters 11-20`…

**Đẩy notebook bằng dòng lệnh** (không bắt buộc): sửa `kaggle/kernel-metadata.json` (tên tài khoản, dataset) rồi:

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
```

## 6. Nghe kiểm tra trước khi đăng

- `truyen.audiblez/qa_report.txt` liệt kê các câu vẫn chưa đạt sau mọi lần thử (đọc sai chữ, mất đuôi câu, giọng
  khác, tốc độ bất thường). Nghe đúng các câu đó trong video.
- Sửa văn bản hoặc từ điển rồi chạy lại: **chỉ câu bị sửa được tạo lại**, phần còn lại lấy từ cache.
  `--retry-failed` tạo lại các câu trong báo cáo mà không cần sửa gì.
- Giọng quá "kịch" hoặc thất thường: `--temperature 0.7` (ổn định hơn, ít biểu cảm hơn; mặc định của model là 0.9).

### Nghe lại lần hai bằng Whisper lớn (`--review`, bật sẵn trong notebook: `REVIEW = True`)

Bộ kiểm tra trong lúc tạo giọng dùng Whisper nhỏ và nới tay để không tạo lại câu chỉ vì Whisper viết tên riêng
bằng chữ Hán khác. Vì vậy nó có thể bỏ sót một từ đọc sai, thậm chí một câu bị nuốt mất (truyện đầu tiên:
「生意気なこと言うな。」 bị bỏ qua). `--review` nghe lại **mọi câu** bằng Whisper large-v3-turbo, so sánh theo
cách đọc (kana, đã áp dụng từ điển của bạn), rồi tạo **4 bản đọc khác** cho các câu đáng ngờ và giữ bản Whisper
nghe rõ nhất. Báo cáo `truyen.audiblez/review.txt` liệt kê phần còn lại:

- từ đọc sai thật → thêm một dòng vào từ điển rồi chạy lại (chỉ câu đó được tạo lại);
- Whisper viết tên riêng khác đi (高円寺 → 公園寺, 源蔵 → 玄蔵): bỏ qua, giọng đọc đã đúng nhờ từ điển.

Một câu chỉ được thử lại một lần (ghi trong `review.json`); chạy lại lệnh chỉ nghe lại những câu đã thay đổi.
Trên GPU mất vài phút; trên CPU khoảng bằng độ dài audio.

### Phụ đề, thẻ chương, màn hình kết thúc, âm thanh nền

- **Phụ đề** `truyen.srt` được tạo cạnh video: đúng chữ gốc (chữ Hán, không phải kana đưa cho giọng đọc), từng
  câu, khớp với chỗ ngắt hơi. Tải lên ở *YouTube Studio → Phụ đề → Thêm ngôn ngữ (Tiếng Nhật) → Tải tệp lên → Có
  thời gian*. Phụ đề chuẩn giúp người xem và giúp YouTube hiểu nội dung (tìm kiếm).
- `--chapter-cards` (`CHAPTER_CARDS = True`): tên chương hiện ở cuối khung hình trong lúc đọc chương đó.
- Video kết thúc bằng 10 giây ảnh bìa làm mờ để gắn *Màn hình kết thúc* của YouTube (video tiếp theo, nút đăng ký);
  `--end-screen 0` để bỏ.
- `--ambience bien.flac` (`AMBIENCE = ...`): một bản thu (sóng biển, mưa, ve sầu…) lặp nhỏ bên dưới giọng đọc,
  vào/ra từ từ. Mặc định `--ambience-level -26` (rất nhẹ; -20 là nghe rõ). Chỉ dùng bản thu bạn có quyền đăng
  (CC0 hoặc tự thu).

## 7. Chạy trên máy có GPU NVIDIA riêng

```bash
sudo apt install ffmpeg
python3.12 -m venv venv && source venv/bin/activate
pip install "audiblez[fast,asr,ja] @ git+https://github.com/vansyson1308/audiblez.git"
audiblez truyen.md --youtube --episode-hours 2 --lexicon tu_dien.txt
```

`[fast]` là backend CUDA graph (chỉ GPU NVIDIA). Trên CPU/Mac dùng `[qwen]` thay cho `[fast]` (chậm: khoảng 8 giờ máy
cho 1 giờ audio trên CPU 4 nhân, chỉ hợp để thử). `audiblez truyen.md --benchmark` đo tốc độ trên máy của anh.
