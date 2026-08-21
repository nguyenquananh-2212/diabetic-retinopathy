# Tài liệu `preprocess_fundus.py`

## 1. Mục đích

Script tiền xử lý ảnh đáy mắt theo từng file. Ảnh gốc không bị thay đổi; ảnh sau xử lý được lưu sang folder mới.

Pipeline:

```text
EXIF orientation → RGB → crop viền tối → giữ tỉ lệ + padding
→ resize vuông 224×224 → lưu PNG
```

Script không thực hiện augmentation ngẫu nhiên. Flip, rotation, brightness và các phép biến đổi ngẫu nhiên sẽ được thực hiện trong `DataLoader` khi train.

Script xử lý một ảnh mỗi lần, lưu ngay kết quả và giải phóng ảnh trước khi chuyển sang file tiếp theo.

## 2. Cấu trúc thư mục

Input mặc định:

```text
D:\Fundus_Project\Data\split_dataset
├── train\0 ... 4
├── val\0 ... 4
└── test\0 ... 4
```

Output mặc định:

```text
D:\Fundus_Project\Data\processed\fundus_224_pad_v1
├── train\0 ... 4
├── val\0 ... 4
├── test\0 ... 4
├── manifest.csv
├── preprocessing_v1.json
└── summary.json
```

Ảnh output được lưu thành PNG để hạn chế mất chi tiết do nén JPEG.

## 3. Hằng số và thư viện

### Thư viện

- `argparse`: đọc tham số dòng lệnh.
- `csv`: ghi thông tin từng ảnh vào `manifest.csv`.
- `json`: ghi cấu hình và thống kê.
- `Counter`: đếm trạng thái xử lý.
- `datetime`, `timezone`: ghi thời gian chạy theo UTC.
- `Path`: xử lý đường dẫn.
- `numpy`: tạo mặt nạ vùng nền tối nhanh hơn vòng lặp Python.
- `PIL.Image`: mở và lưu ảnh.
- `PIL.ImageFile`: cho phép cố gắng đọc ảnh truncate.
- `PIL.ImageOps`: sửa hướng EXIF và padding ảnh.

`import os` hiện đang được khai báo nhưng chưa sử dụng.

### Hằng số

```python
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_SPLITS = ("train", "val", "test")
```

`SUPPORTED_EXTENSIONS` giới hạn loại file được xử lý. `EXPECTED_SPLITS` quy định các folder dataset cần duyệt.

```python
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

Cho phép PIL cố gắng đọc một số ảnh bị thiếu dữ liệu cuối file. Ảnh không đọc được vẫn được ghi lỗi vào manifest.

## 4. Giải thích từng hàm

### `parse_args()`

```python
def parse_args() -> argparse.Namespace:
```

Đọc các tùy chọn khi chạy script.

```python
project_root = Path(__file__).resolve().parents[1]
```

Tìm thư mục gốc project từ vị trí file script. Với project hiện tại, kết quả là `D:\Fundus_Project`.

Các tham số:

| Tham số | Mặc định | Chức năng |
|---|---|---|
| `--input-root` | `Data/split_dataset` | Folder ảnh gốc |
| `--output-root` | `Data/processed/fundus_224_pad_v1` | Folder ảnh output |
| `--size` | `224` | Kích thước output, ví dụ `224×224` |
| `--dark-threshold` | `10` | Ngưỡng xác định pixel nền tối |
| `--crop-margin` | `0.02` | Biên phụ quanh vùng foreground |
| `--overwrite` | Tắt | Xử lý lại file output đã tồn tại |

Hàm trả về một `argparse.Namespace`, có thể truy cập bằng `args.size`, `args.input_root` và các thuộc tính tương tự.

### `resolve_path(path)`

```python
def resolve_path(path: Path) -> Path:
```

Chuyển path tương đối thành path tuyệt đối:

```python
return path.expanduser().resolve()
```

Hàm giúp script làm việc ổn định dù người dùng chạy lệnh từ folder nào.

### `is_inside(child, parent)`

```python
def is_inside(child: Path, parent: Path) -> bool:
```

Kiểm tra `child` có nằm bên trong `parent` hay không bằng `child.relative_to(parent)`.

Hàm được dùng để ngăn output nằm trong input. Nếu không kiểm tra, lần chạy sau có thể đọc lại ảnh đã xử lý.

### `image_files(root)`

```python
def image_files(root: Path) -> Iterable[Path]:
```

Tìm file ảnh đệ quy bằng `root.rglob("*")`, sắp xếp kết quả và chỉ trả về các phần mở rộng nằm trong `SUPPORTED_EXTENSIONS`.

Hàm dùng `yield`, nên trả về từng path thay vì nạp nội dung toàn bộ ảnh vào RAM.

### `crop_dark_border(image, dark_threshold, margin_fraction)`

```python
def crop_dark_border(
    image: Image.Image,
    dark_threshold: int,
    margin_fraction: float,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
```

Loại bỏ viền đen ngoài vùng võng mạc nhưng không crop cố định ở giữa ảnh.

#### Chuyển sang RGB và NumPy

```python
rgb = image.convert("RGB")
pixels = np.asarray(rgb, dtype=np.uint8)
```

Ảnh trở thành mảng có dạng `height × width × 3`.

#### Tạo mặt nạ foreground

```python
foreground = pixels.max(axis=2) > dark_threshold
```

Nếu giá trị lớn nhất trong ba kênh RGB của pixel lớn hơn ngưỡng, pixel được xem là vùng ảnh thật. Với mặc định `10`, pixel có độ sáng tối đa không quá 10 được xem là nền.

#### Tìm bounding box

```python
ys, xs = np.where(foreground)
min_x, max_x = int(xs.min()), int(xs.max())
min_y, max_y = int(ys.min()), int(ys.max())
```

Tìm hình chữ nhật nhỏ nhất bao quanh toàn bộ foreground.

Nếu không có foreground, hàm trả ảnh gốc. Nếu foreground chiếm ít nhất khoảng 98% chiều rộng và chiều cao, hàm cũng không crop để tránh crop do nhiễu.

#### Thêm margin và crop

```python
margin_x = max(1, int(box_width * margin_fraction))
margin_y = max(1, int(box_height * margin_fraction))
```

Với `crop_margin=0.02`, thêm 2% biên quanh vùng phát hiện được. Sau đó tọa độ được giới hạn trong kích thước ảnh bằng `max()` và `min()`.

Hàm trả về:

```python
(cropped_image, (left, top, right, bottom))
```

Tọa độ crop được ghi lại vào manifest để kiểm tra.

### `pad_and_resize(image, size)`

```python
def pad_and_resize(image: Image.Image, size: int) -> Image.Image:
```

Đưa ảnh về hình vuông mà không làm méo tỉ lệ:

```python
ImageOps.pad(
    image.convert("RGB"),
    (size, size),
    method=Image.Resampling.LANCZOS,
    color=(0, 0, 0),
    centering=(0.5, 0.5),
)
```

- `LANCZOS`: nội suy chất lượng cao khi resize.
- `color=(0, 0, 0)`: màu padding.
- `centering=(0.5, 0.5)`: đặt ảnh ở giữa canvas.

Khác với `Resize((224, 224))`, hàm này không kéo giãn ảnh `2588×1958` thành hình vuông một cách méo mó.

### `output_name(source_path, output_dir)`

```python
def output_name(source_path: Path, output_dir: Path) -> Path:
```

Tạo đường dẫn output, đổi phần mở rộng thành `.png`.

Ví dụ:

```text
0da321efbce6-600.jpg → 0da321efbce6-600.png
```

Nếu tên output đã tồn tại, hàm thêm phần mở rộng gốc vào tên, ví dụ `image__jpg.png`, để hạn chế xung đột tên.

### `process_one(...)`

```python
def process_one(
    source_path: Path,
    output_dir: Path,
    size: int,
    dark_threshold: int,
    crop_margin: float,
    overwrite: bool,
) -> Dict[str, object]:
```

Xử lý một ảnh từ đầu đến cuối.

#### Tạo folder và bản ghi

```python
output_dir.mkdir(parents=True, exist_ok=True)
```

Tạo folder output nếu chưa tồn tại.

Dictionary `row` lưu đường dẫn gốc, đường dẫn output, trạng thái, kích thước gốc, tọa độ crop và lỗi.

#### Bỏ qua output đã tồn tại

```python
if destination.exists() and not overwrite:
    row["status"] = "skipped_exists"
```

Mặc định script không xử lý lại output đã có. Truyền `--overwrite` nếu muốn ghi lại.

#### Mở ảnh và sửa EXIF

```python
with Image.open(source_path) as opened:
    image = ImageOps.exif_transpose(opened).convert("RGB")
```

Context manager đóng file nguồn sau khi đọc. `exif_transpose` sửa hướng ảnh theo metadata EXIF trước khi chuyển sang RGB.

#### Chạy pipeline và lưu ảnh

```python
cropped, crop_box = crop_dark_border(image, dark_threshold, crop_margin)
normalized = pad_and_resize(cropped, size)
normalized.save(destination, format="PNG", optimize=True)
```

Nếu có lỗi, lỗi được ghi vào `row["error"]`; script không dừng toàn bộ dataset.

Hàm trả về `row` để `main()` ghi thành một dòng trong `manifest.csv`.

### `write_json(path, payload)`

```python
def write_json(path: Path, payload: Dict[str, object]) -> None:
```

Ghi dictionary thành JSON:

```python
json.dumps(payload, indent=2, ensure_ascii=False)
```

`indent=2` giúp file dễ đọc; `ensure_ascii=False` giữ được ký tự Unicode.

Hàm được dùng cho `preprocessing_v1.json` và `summary.json`.

### `main()`

```python
def main() -> int:
```

Đây là hàm điều phối toàn bộ chương trình.

Các bước chính:

1. Gọi `parse_args()` để đọc tham số.
2. Kiểm tra `size`, `dark_threshold` và `crop_margin`.
3. Chuẩn hóa input/output bằng `resolve_path()`.
4. Kiểm tra input tồn tại.
5. Kiểm tra output không nằm bên trong input bằng `is_inside()`.
6. Tạo output folder.
7. Ghi cấu hình vào `preprocessing_v1.json`.
8. Mở `manifest.csv` và ghi header.
9. Duyệt lần lượt `train`, `val`, `test`.
10. Duyệt từng class folder và gọi `process_one()` cho từng ảnh.
11. Ghi manifest và `flush()` ngay sau mỗi ảnh.
12. In tiến độ sau mỗi 500 ảnh.
13. Ghi thống kê cuối cùng vào `summary.json`.

`main()` trả về `0` khi hoàn tất bình thường.

## 5. Khối chạy trực tiếp

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Khối này chỉ gọi `main()` khi chạy trực tiếp:

```powershell
python .\Source\preprocess_fundus.py
```

Nếu import file như một module, `main()` không tự động chạy.

## 6. Cách chạy

### Cấu hình mặc định

```powershell
python .\Source\preprocess_fundus.py
```

### Resize sang 300×300 và lưu phiên bản khác

```powershell
python .\Source\preprocess_fundus.py `
  --size 300 `
  --output-root .\Data\processed\fundus_300_pad_v1
```

### Xử lý lại và ghi đè

```powershell
python .\Source\preprocess_fundus.py --overwrite
```

### Đổi ngưỡng nền tối

```powershell
python .\Source\preprocess_fundus.py --dark-threshold 15
```

Chỉ nên đổi ngưỡng sau khi xem một số ảnh output; ngưỡng quá cao có thể crop nhầm vùng tối của võng mạc.

## 7. Các file metadata

### `manifest.csv`

Lưu một dòng cho mỗi ảnh với các cột:

```text
split, class_label, source_path, processed_path, status,
original_width, original_height,
crop_left, crop_top, crop_right, crop_bottom, error
```

`status` thường là `processed`, `skipped_exists` hoặc `error`.

### `preprocessing_v1.json`

Lưu cấu hình đã dùng, chẳng hạn kích thước output, ngưỡng crop, định dạng ảnh và việc augmentation có được thực hiện hay không.

ImageNet mean/std không được ghi trực tiếp lên PNG. Phần đó sẽ áp dụng trong `DataLoader` cho Swin Transformer và EfficientNetV2-S.

### `summary.json`

Lưu tổng số entry và số lượng theo từng trạng thái xử lý.

## 8. Kiểm tra sau khi chạy

Kiểm tra `summary.json`, các dòng `status=error` trong `manifest.csv` và kích thước ảnh output.

```python
from pathlib import Path
from PIL import Image

root = Path("Data/processed/fundus_224_pad_v1")

for path in root.rglob("*.png"):
    with Image.open(path) as image:
        assert image.size == (224, 224), (path, image.size)

print("All processed images are 224x224")
```

