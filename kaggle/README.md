# Render truyện audio trên Kaggle miễn phí (GPU T4 × 2)

Kaggle cho mỗi tài khoản đã xác minh số điện thoại khoảng **30 giờ GPU mỗi tuần** (reset 00:00 UTC thứ Bảy), mỗi phiên
tối đa 12 giờ, với **hai GPU T4 16 GB**. Notebook `audiblez_kaggle.ipynb` dùng cả hai GPU cùng lúc và render **mọi truyện**
trong danh sách `BOOK`, rồi tự dừng gọn trước khi hết phiên (`--time-limit`).

## 1. Chuẩn bị dataset đầu vào (riêng tư)

```
audiblez-input/
├── 2026-11-25_bran_na_spor.txt        ← các truyện: .txt .md .docx .fb2 .epub
├── 2026-11-26_svekrov.md
├── voices/giong_nu/reference.flac     ← (tuỳ chọn) giọng riêng: 6-15 s + transcript chính xác
├── voices/giong_nu/reference.txt
├── tu_dien.txt                        ← (tuỳ chọn) từ điển phát âm, mỗi dòng "từ = cách đọc"
└── cover.jpg                          ← (tuỳ chọn) ảnh nền video
```

Trên web: kaggle.com → Datasets → New Dataset → kéo thả thư mục → **Private**. Lần sau thêm truyện mới: *New Version*.

## 2. Tạo notebook

**Cách web (đơn giản nhất)**: Code → New Notebook → File → *Import Notebook* → chọn `kaggle/audiblez_kaggle.ipynb`.
Ở khung bên phải: *Add Input* → dataset vừa tạo; *Session options* → **Accelerator: GPU T4 x2**, **Internet: On**.
Sửa ô *Settings / Cấu hình*: `BOOK` (đường dẫn trong `/kaggle/input/...`, hoặc danh sách nhiều truyện, vd.
`BOOK = sorted(glob.glob('/kaggle/input/audiblez-input/*.txt'))` sau `import glob`),
`VOICE` (`'ru_female_narrator'`, `'ru_male_narrator'` hoặc thư mục giọng trong dataset, vd.
`'/kaggle/input/audiblez-input/voices/giong_nu'`), `STRESS = True` (trọng âm), `EXTRA` (vd.
`'--lexicon /kaggle/input/audiblez-input/tu_dien.txt --slides /kaggle/input/audiblez-input/slides.txt'`), rồi
**Save Version → Save & Run All (Commit)**:
notebook chạy nền đến hết, có thể tắt trình duyệt. Kết quả ở tab *Output* của version đó.

**Cách dòng lệnh** (máy cá nhân, `pip install kaggle`, token API ở `~/.kaggle/kaggle.json`):

```bash
kaggle datasets init -p audiblez-input          # sửa id/title trong dataset-metadata.json
kaggle datasets create -p audiblez-input -r zip
kaggle datasets version -p audiblez-input -r zip -m "truyện mới"   # các lần sau
```

Sửa `kaggle/kernel-metadata.json` (tên tài khoản Kaggle của anh trong `id`, dataset trong `dataset_sources`), rồi:

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status YOUR_KAGGLE_USERNAME/audiblez-render
kaggle kernels output YOUR_KAGGLE_USERNAME/audiblez-render -p ket_qua
```

**Bật Internet** cho notebook: model Qwen3-TTS (~4,5 GB), Whisper và RUAccent được tải từ Hugging Face ở đầu mỗi phiên
(vài phút). Không dùng `HF_HUB_OFFLINE=1` với model lấy theo tên: thư viện transformers 4.57 vẫn gọi mạng khi nạp
tokenizer của Qwen3-TTS và báo lỗi; muốn chạy không mạng thì đưa model vào dataset và truyền đường dẫn thư mục bằng
`--qwen-model /kaggle/input/...`.

## 3. Lần chạy đầu: đo trước, render sau

Để `RUN_BENCHMARK = True` cho lần đầu (~10–15 phút): audiblez đo các độ chính xác số, backend và batch trên T4 với
chính truyện và giọng của anh, chỉ giữ cấu hình mà mọi câu qua kiểm tra chất lượng, in dòng
`Fastest: --backend … --batch-size … --dtype …` và notebook tự dùng nó cho bước render. Ghi lại vào
`BACKEND, BATCH, DTYPE` rồi đặt `RUN_BENCHMARK = False` cho các lần sau.

Backend `faster` (CUDA graphs) tự chuyển về backend gốc nếu gặp lỗi, nên một phiên dài không bị mất.

## 4. Kết quả

Trong `output/`: `<truyện>.mp4` (upload YouTube), `<truyện>_youtube.txt` (mốc thời gian cho mô tả), `<truyện>.m4b`,
và thư mục `<truyện>.audiblez/` (cache từng câu, `script.txt`, `qa_report.txt`); cache được giữ để làm lại câu lỗi.

**Phiên bị dừng giữa chừng** (hết 12 giờ, hoặc truyện chưa xong): thêm Output của lần chạy trước làm Input của notebook
(*Add Input → Notebook Output*) và đặt `RESUME_FROM` tới thư mục `output` của nó; notebook chép cache về và chỉ tổng
hợp phần còn thiếu.

**Câu lỗi trong `qa_report.txt`**: nghe thử; nếu cần làm lại, chạy trên máy nhà trong thư mục output đã tải về:
`audiblez truyen.txt -o ket_qua --voice-profile ... --retry-failed` (cache dùng chung giữa backend faster và gốc).

## 5. Quy tắc nên giữ

- **Một người một tài khoản**: lập nhiều tài khoản để nhân quota vi phạm điều khoản Kaggle. Cách cộng dồn hợp lệ là
  dùng thêm nền tảng khác (Modal, GPU thuê theo giờ) — audiblez chạy giống hệt ở mọi nơi.
- Upload video lên YouTube từ máy cá nhân, không từ notebook.
- Đo bằng benchmark trước khi tin bất kỳ con số tốc độ nào: số 0,93x thời gian thực mỗi T4 của faster-qwen3-tts là do
  tác giả tự đo.
