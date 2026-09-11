# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** `11/09/2026`

**Runtime Colab:** `GPU T4`

**Python / PyTorch / Ultralytics:** `Python 3.11`, `PyTorch 2.5.1`, `Ultralytics 8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):{
  "taxonomy_name": "ImageNet-1K",
  "rank": 1,
  "class_id": 468,
  "class_name": "cab",
  "score": 0.510915
  }
- Record này mô tả toàn ảnh như thế nào?
  > Record này chỉ gán duy nhất một nhãn tổng quát cho toàn bộ bức ảnh ("cab"). Nó không xác định được vị trí của xe taxi nằm ở tọa độ nào, không đếm được có bao nhiêu chiếc xe, và hoàn toàn bỏ qua các đối tượng khác trong ảnh (như xe buýt lớn, xe con, người đi bộ).
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  > Class list được định nghĩa bởi bộ dữ liệu ImageNet-1K gồm 1.000 lớp định nghĩa sẵn từ trước, thường là các chuyên gia trong lĩnh vực nhận diện hình ảnh.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  > Việc giữ cả ba thông tin này giúp đảm bảo tính chính xác và khả năng tái sử dụng của dữ liệu. ID giúp xác định duy nhất mỗi lớp, tên cal cung cấp ngữ nghĩa rõ ràng, còn tên taxonomy giúp phân loại và tổ chức các lớp theo ImageNet-1K.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  > Guideline cần quy định cách xử lý khi ảnh có nhiều chủ thể, ví dụ như chọn chủ thể chính hoặc xử lý từng chủ thể riêng biệt.
- Vì sao model score không phải ground truth?
  > Model score chỉ là giá trị xác suất toán học do mô hình tự ước tính dựa trên trọng số đã học (mô hình đoán "tôi tự tin 51% đây là xe taxi"). Mô hình hoàn toàn có thể tự tin nhưng phán đoán sai (trong ảnh đối tượng nổi bật rõ rệt nhất là xe buýt đô thị lớn). Ground truth phải là sự thật khách quan do con người kiểm chứng và xác nhận.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  {
  "class_name": "person",
  "score": 0.769676,
  "bbox_xyxy": [
  0.08,
  256.79,
  18.39,
  313.12
  ],
  "bbox_width": 18.32,
  "bbox_height": 56.33
  }
- Diễn giải vị trí box bằng lời:
  > Box này xác định một đối tượng "person" (người) trong ảnh, với tọa độ góc trên bên trái là (0.08, 256.79) và góc dưới bên phải là (18.39, 313.12). Chiều rộng của box là 18.32 pixel và chiều cao là 56.33 pixel. Hộp này bao trọn lấy người đầu bếp từ đầu tới chân, đang mặc áo trắng và quay lưng đứng ở nửa bên phải gian bếp.
- So sánh số prediction ở hai threshold:
  > Ở ngưỡng chuẩn threshold = 0.35: Mô hình phát hiện được 11 vật thể (person, bowl, oven, cup...). Nếu nâng ngưỡng lên cao threshold = 0.60: Chỉ còn giữ lại 6 vật thể có độ tự tin cao (2 person, 2 bowl, 2 oven). Các vật thể nhỏ hoặc mờ như cup hay bowl nhỏ bị loại bỏ hoàn toàn
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  > Khi hạ threshold (ngưỡng thấp): Độ bao phủ (Recall) tăng, bắt trúng nhiều vật thể thật hơn và ít bị sót, nhưng mô hình sẽ bắt nhầm nhiều rác (False Positive). Khi đó, khối lượng công việc của reviewer tăng vọt vì phải tốn công ngồi xóa các box sai. Khi nâng threshold (ngưỡng cao): Reviewer ít việc hơn vì mô hình chỉ đưa ra các box rất chắc chắn, nhưng nguy cơ bỏ sót vật thể thực tế (False Negative) vẫn là rất lớn.
- Đề xuất một quy tắc box chặt:
  > Bounding box phải ôm sát 4 mép ngoài cùng của vật thể (kể cả phần nhô ra như tay, quai cầm), lề thừa không được quá 2–3 pixel so với điểm cực biên, và tuyệt đối không được cắt lẹm vào chi tiết của vật thể.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  > Guideline cần quy định rõ ràng cách xử lý các vật thể bị che khuất hoặc cắt mép. Vật thể bị che khuất/cắt mép bao nhiêu % thì được phép gán nhãn (ví dụ: thấy trên 20% thì vẽ box, dưới 20% bỏ qua). Nếu gặp trường hợp lấp lửng không có trong hướng dẫn (như một cánh tay rời ở mép ảnh), cần phải báo cáo cấp trên để thống nhất quy chuẩn, không được tự ý phán đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  {
  "instance_id": "kitchen-008",
  "class_name": "spoon",
  "score": 0.487577,
  "polygon_point_count": 46,
  "polygon_xy": [
  [
  544.0,
  67.0
  ],
  [
  544.0,
  80.0
  ],
  [
  545.0,
  81.0
  ],
  [
  545.0,
  91.0
  ],
  [
  544.0,
  92.0
  ],
  [
  544.0,
  101.0
  ],
  ...
  ]
  },
- Polygon bổ sung chi tiết gì so với box?
  > Polygon cung cấp một đường viền chính xác hơn bao quanh vật thể, cho phép mô tả hình dạng thực tế của đối tượng (như tay cầm của thìa) thay vì chỉ là một hình chữ nhật đơn giản. Điều này giúp giảm phần nền thừa và tăng độ chính xác trong việc phân đoạn.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  > `instance_id` dùng để phân biệt các vật thể riêng lẻ trong cùng một lớp (ví dụ: hai chiếc thìa khác nhau trong cùng một bức ảnh). Nó không phải là ID của lớp hay nhãn, mà là một định danh duy nhất cho từng instance cụ thể.
- Đề xuất một quy tắc biên mask:
  > Biên mask đa giác phải bám khít biên dạng thực của vật thể với sai số không vượt quá 2 pixel; không được khoét phạm vào thân vật thể và không được để lọt viền nền xung quanh; các khúc cong phải đặt đủ mật độ điểm kiểm soát để đường bao trơn tru, không gấp khúc méo mó.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  > Guideline cần quy định rõ ràng cách xử lý các vùng mờ, tiếp xúc hoặc che khuất. Nếu vật thể bị che khuất một phần, cần xác định tỷ lệ phần bị che khuất để quyết định có vẽ mask hay không (ví dụ: nếu trên 30% thì vẽ mask, dưới 30% bỏ qua). Nếu gặp trường hợp khó phân biệt hoặc không có trong hướng dẫn, cần báo cáo cấp trên để thống nhất quy chuẩn, tránh tự ý phán đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Đơn vị/định dạng ground truth                                     | Lỗi hoặc điểm mơ hồ quan sát được                                                                                                        | Annotator làm gì?                                                                                                                                                                  | Reviewer xem gì?                                                                                                                                     |
| --------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Nhãn duy nhất cấp ảnh (`class_id`, `class_name`) theo ImageNet-1K | Ảnh`traffic` có nhiều đối tượng phức tạp (xe buýt lớn, ô tô con, người đi bộ) nhưng mô hình chỉ đoán nhãn đơn `cab` (0.51)               | Nghiên cứu guideline để chọn nhãn đại diện theo quy tắc ưu tiên: chọn chủ thể chiếm diện tích lớn nhất ở trung tâm hoặc nhãn bối cảnh bao quát; gắn cờ mơ hồ nếu cảnh quá phức tạp | Kiểm tra nhãn được gán có tuân thủ đúng taxonomy ImageNet-1K và nguyên tắc phân cấp ưu tiên chủ thể trong guideline hay không                        |
| Phát hiện vật thể     | Tập hợp các cặp`(class_id, bbox_xyxy)` cho từng vật thể           | Bàn tay ở mép trái ảnh`kitchen` bị gắn nhãn `person` (0.61); mô hình bỏ sót nhiều xoong chảo trên giá và nhận nhầm khuôn nướng là `bowl` | Vẽ bounding box ôm sát 4 điểm cực biên của vật thể; kiểm tra ngưỡng diện tích nhìn thấy (>20%); đánh dấu cờ`truncated` cho vật thể bị cắt mép                                      | Soi độ khít (tightness) của box (loại bỏ box quá lỏng hoặc cắt lẹm); rà soát kỹ các vật thể bị bỏ sót; kiểm tra tính chính xác của nhãn lớp          |
| Instance segmentation | Tập hợp`(instance_id, class_id, polygon_xy)` cho từng cá thể      | Bó lá khô treo trên tường bị nhận nhầm là`potted plant` (0.63); mặt nạ bàn bếp lớn bị đục khoét lởm chởm do bát đĩa đè lên               | Dùng công cụ đa giác viền khít biên dạng thực của từng cá thể; xử lý đục lỗ mặt nạ đúng theo quy định khi có đối tượng đè lên; tách riêng từng instance                            | Phóng to kiểm tra độ chính xác đường biên (pixel accuracy), phát hiện các vùng lem ra nền hoặc ăn phạm vào vật thể; duyệt các vùng tiếp xúc phức tạp |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  > Tuyệt đối không tải lên kho lưu trữ công khai (GitHub public / Colab) các hình ảnh riêng tư, khuôn mặt chưa che mờ, biển số xe, thông tin định danh cá nhân (PII như họ tên, MSSV, SĐT) hoặc dữ liệu nội bộ/bí mật kinh doanh. Chỉ sử dụng dữ liệu mở có giấy phép công khai (như COCO CC BY 2.0).
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  > Giảng viên hướng dẫn (GV) / Lab Coach / Người phụ trách dự án qua kênh hỗ trợ chính thức của lớp để cách ly và xử lý dữ liệu kịp thời.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
