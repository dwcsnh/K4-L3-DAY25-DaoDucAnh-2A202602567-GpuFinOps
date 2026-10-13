# NimbusAI — Báo Cáo Kỹ Thuật FinOps (Write-up)

## 1. Baseline vs. Optimized
Chi phí GPU ban đầu của NimbusAI theo dữ liệu ghi nhận là $27,133/tháng (trong đó $1,475.25 đến từ hoạt động inference và $25,658 từ các tác vụ training, batch và dev trên workloads.csv). Khi áp dụng phối hợp các đòn bẩy tối ưu hóa, mức chi phí ước tính giảm xuống khoảng $14,904/tháng theo chính sách có tính đến rủi ro thu hồi spot, tương đương mức giảm dự kiến $12,229/tháng (khoảng 45.1% ngân sách GPU).

Về mặt đơn vị kinh tế (unit economics), chi phí trên một triệu token ($/1M-token) trong mô phỏng giảm từ $6.488/1M-token xuống còn $1.126/1M-token, tương đương mức giảm khoảng 82.6%. Điều này phản ánh hiệu quả sử dụng tài nguyên được cải thiện, giúp phục vụ được nhiều token hơn với cùng một mức chi phí bỏ ra.

## 2. Phân tích từng đòn bẩy
Cơ cấu tiết kiệm ước tính theo từng nhóm đòn bẩy gồm:
- Chiến lược mua sắm (Purchasing Strategy - Spot / Reserved): khoảng $9,762/tháng (chiếm phần lớn tổng mức tiết kiệm).
- Kỹ thuật tối ưu suy luận (Inference Levers - Cascade, Prompt Caching, Batch API): khoảng $1,212/tháng.
- Điều chỉnh kích thước GPU (Right-sizing theo MFU/MBU): khoảng $655/tháng.
- Giảm thiểu GPU nhàn rỗi (Kill idle GPUs): khoảng $600/tháng.

Đòn bẩy đóng góp tỷ trọng lớn nhất là chiến lược mua sắm (Spot và Reserved). Nguyên nhân chủ yếu do các tác vụ huấn luyện và xử lý lô của đội ngũ kỹ thuật có quy mô sử dụng giờ GPU khá lớn (chẳng hạn job-train-llm chạy 8 GPU H100 trong 20 giờ mỗi ngày, hay job-infer-chat dùng 6 GPU A10G chạy liên tục). Việc chuyển các job huấn luyện có checkpoint sang spot và cam kết reserved cho các cụm inference ổn định giúp tận dụng mức chiết khấu sẵn có của nhà cung cấp cloud trên khối lượng lớn giờ máy.

## 3. GPU-Util Lie
Trong quá trình kiểm toán M1, hệ thống ghi nhận hai trường hợp gpu-h100-4 và gpu-a10g-1 có hiện tượng chênh lệch giữa mức tải bề mặt và hiệu năng thực, rõ nhất là gpu-h100-4.
Trên nvidia-smi, gpu-h100-4 hiển thị mức sử dụng lên tới 98.2%. Tuy nhiên, Model FLOPs Utilization (MFU) tính toán được chỉ ở mức 0.194 (19.4%) và Model Bandwidth Utilization (MBU) là 0.207.

Nguyên nhân kỹ thuật: Thông số GPU-Util trên nvidia-smi phản ánh tỷ lệ thời gian mà vi xử lý GPU có phát sinh xung nhịp (time-active clock), chứ không phản ánh lượng tính toán hữu ích thực tế. Khi workload gặp tình trạng nghẽn truyền dữ liệu bộ nhớ, kích thước batch chưa tối ưu hoặc nạp dữ liệu từ CPU chậm, GPU vẫn ở trạng thái bận nhưng phần lớn thời gian phải chờ đợi dữ liệu.
Tác động chi phí: Doanh nghiệp chi trả chi phí thuê theo giờ của GPU H100 ($2.50/giờ) nhưng năng lực xử lý nhận lại chỉ ở mức tương đương các dòng GPU thấp hơn như A100 ($1.79/giờ). Sự chênh lệch hiệu quả này gây lãng phí ước tính từ $511 đến $655 mỗi tháng trên mỗi GPU tương tự.

## 4. Các phần mở rộng đã thực hiện (Extensions 1, 2, 3)
### Extension 1 — Cải thiện recommend_tier() theo tỷ lệ thu hồi và thời hạn job
Bổ sung logic phân tích tỷ lệ thu hồi (preemption rate) theo dòng phần cứng: H100 có tỷ lệ thu hồi tương đối thấp (~3%/giờ), phù hợp hơn để cân nhắc chạy spot khi có cơ chế lưu checkpoint định kỳ. Trong khi đó, A10G ghi nhận tỷ lệ thu hồi cao hơn (~20%/giờ). Đối với các môi trường dev mang tính tương tác như job-dev-sandbox, việc bị ngắt giữa chừng có thể làm gián đoạn công việc của kỹ sư. Do đó, chính sách đề xuất giữ job-dev-sandbox ở on-demand. Mức tiết kiệm trên giấy giảm nhẹ từ 39.1% xuống 38.0%, nhưng giúp hạn chế rủi ro gián đoạn công việc thực tế. Đồng thời, hàm hỗ trợ đánh giá thời hạn job để cân nhắc giữa hợp đồng 1 năm và 3 năm nhằm hạn chế rủi ro cam kết dài hạn khi nhu cầu chưa ổn định.

### Extension 2 — Right-sizing theo MBU và $/GB-VRAM
Xây dựng bảng tổng hợp chi phí bộ nhớ trên danh mục GPU: MI300X đạt $0.0102/GB-hr, A100 đạt $0.0224/GB-hr, H100 đạt $0.0312/GB-hr, và A10G là $0.0417/GB-hr. Với các GPU memory-bound có MBU dưới 0.32, thuật toán so sánh băng thông thực tế (achieved bandwidth) với năng lực của các dòng GPU khác: gpu-h100-4 (băng thông thực tế 0.69 TB/s) có thể cân nhắc chuyển sang A100 (băng thông tối đa 2.0 TB/s), còn gpu-a10g-0 và gpu-a10g-1 có thể chuyển sang L4 (0.3 TB/s). Mức tiết kiệm ước tính có thể đạt thêm khoảng $1,310/tháng nếu chuyển đổi thành công.
Bài học rút ra: Không nên chỉ nhìn vào mức giá $/GPU-giờ thấp nhất để lựa chọn. Nếu chọn GPU rẻ nhưng băng thông bộ nhớ không đủ đáp ứng nhu cầu xử lý của mô hình, hệ thống sẽ gặp hiện tượng nghẽn cổ chai, làm tăng độ trễ và có thể ảnh hưởng tiêu cực đến chất lượng dịch vụ.

### Extension 3 — Đo lường hòa vốn Prompt Caching với cache_is_worth_it()
Xây dựng công thức ước tính điểm hòa vốn: Khi chi phí ghi cache bằng 1.25 lần giá input thông thường và mức giảm giá đọc là 90%, điểm hòa vốn cần đạt khoảng 1.39 lượt đọc (khoảng 2 lần đọc lại prefix). Trong tập dữ liệu mẫu token_usage.csv với 2,400 request chia sẻ system prompt của 4 dự án, tần suất tái sử dụng trung bình đạt khoảng 25 lần mỗi prefix. Vì con số này cao hơn ngưỡng hòa vốn 1.39, việc áp dụng Prompt Caching trong kịch bản này là tương đối hợp lý và có thể mang lại hiệu quả giảm chi phí token đầu vào.

## 5. Khuyến nghị cho NimbusAI
Dưới góc độ FinOps, một số định hướng đề xuất có thể ưu tiên xem xét triển khai:
1. Thử nghiệm áp dụng mô hình Spot kết hợp Reserved theo từng nhóm workload: Cân nhắc chuyển các tác vụ huấn luyện có khả năng checkpoint sang spot, đồng thời đánh giá nhu cầu tải ổn định để cân nhắc cam kết reserved cho các dịch vụ inference cốt lõi. Trong kịch bản mô phỏng, phương án này có tiềm năng đóng góp tỷ trọng tiết kiệm lớn nhất (ước tính khoảng $9,762/tháng).
2. Tích hợp Model Cascading và Prompt Caching vào tầng ứng dụng: Thử nghiệm định tuyến các truy vấn đơn giản sang mô hình nhỏ hơn và bật lưu cache các system prompt có tần suất lặp lại cao. Trong dữ liệu mẫu, giải pháp này giúp cải thiện chi phí suy luận từ $6.488 xuống khoảng $1.126 trên mỗi triệu token.
3. Hoàn thiện quy trình phân bổ chi phí và rà soát tài nguyên nhàn rỗi: Với tỷ lệ gắn tag hiện đạt 92% (đáp ứng điều kiện trên 80%), công ty có thể từng bước chuyển từ mô hình showback sang chargeback để nâng cao ý thức sử dụng ngân sách của các nhóm. Song song đó, có thể thiết lập cảnh báo hoặc tự động tắt các máy ảo không phát sinh tải sau một khoảng thời gian nhất định, giúp hạn chế bớt chi phí phát sinh ngoài giờ (ước tính khoảng $600/tháng trong phân tích).
