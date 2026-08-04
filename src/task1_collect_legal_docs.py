"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Các nguồn tài liệu chính thức từ Shopee Vietnam (help.shopee.vn):
    1. https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    2. https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    3. https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)
    4. Quy định đăng bán sản phẩm dành cho người bán (Seller Listing Regulations)

Yêu cầu K4 Variant:
    - Lưu các file PDF/DOCX vào data/landing/legal/
    - Đặt tên file rõ ràng, không dấu
    - Gắn metadata customer_role (buyer/seller/both)
"""

import sys
import os
from pathlib import Path
from fpdf import FPDF

# Đảm bảo stdout in utf-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Danh sách văn bản chính sách cần khởi tạo
POLICIES = [
    {
        "filename": "returns-refund-policy-shopee.pdf",
        "title": "CHÍNH SÁCH TRẢ HÀNG VÀ HOÀN TIỀN SHOPEE VIỆT NAM",
        "url": "https://help.shopee.vn/portal/4/article/77251",
        "customer_role": "buyer",
        "sections": [
            ("1. Điều kiện yêu cầu Trả hàng / Hoàn tiền", 
             "Người mua có quyền gửi yêu cầu trả hàng và hoàn tiền trong vòng 15 ngày kể từ ngày giao hàng thành công (hoặc 24 giờ đối với các mặt hàng thực phẩm tươi sống, thực phẩm đông lạnh). "
             "Các trường hợp được áp dụng bao gồm: Sản phẩm bị lỗi kỹ thuật, hỏng hóc trong quá trình vận chuyển; sản phẩm giao không đúng chủng loại, màu sắc hoặc thiếu số lượng; sản phẩm bị sứt mỡ, biến dạng; sản phẩm nghi ngờ là hàng giả, hàng nhái hoặc không giống mô tả của Người bán."),
            ("2. Quy trình xử lý yêu cầu", 
             "Người mua thực hiện các bước: (1) Mở ứng dụng Shopee -> Chọn Đơn mua -> Chọn Yêu cầu Trả hàng/Hoàn tiền. (2) Chọn lý do phù hợp và tải lên hình ảnh/video bằng chứng mở hộp sản phẩm. (3) Người bán có tối đa 3 ngày để phản hồi. Nếu Người bán đồng ý hoặc không phản hồi quá thời hạn, Shopee sẽ phát mã vận chuyển trả hàng miễn phí. (4) Người mua đóng gói sản phẩm và bàn giao cho đơn vị vận chuyển."),
            ("3. Phương thức và thời gian hoàn tiền", 
             "Sau khi đơn hàng trả về được kiểm tra hợp lệ: "
             "- Hoàn qua Ví ShopeePay: Trong vòng 24 giờ làm việc. "
             "- Hoàn qua Thẻ tín dụng / Thẻ ghi nợ: Từ 7 đến 14 ngày làm việc tùy thuộc vào ngân hàng phát hành. "
             "- Hoàn qua Tài khoản ngân hàng liên kết: Trong vòng 24 đến 48 giờ làm việc."),
            ("4. Các trường hợp từ chối yêu cầu", 
             "Shopee từ chối hoàn tiền đối với: Sản phẩm đã bị bóc tem niêm phong vệ sinh/an toàn, sản phẩm đã qua sử dụng, hư hỏng do Người mua bảo quản sai hướng dẫn, hoặc yêu cầu gửi sau khi quá thời hạn 15 ngày quy định.")
        ]
    },
    {
        "filename": "payment-methods-shopee.pdf",
        "title": "QUY ĐỊNH VỀ CÁC PHƯƠNG THỨC THANH TOÁN TRÊN SHOPEE",
        "url": "https://help.shopee.vn/portal/4/article/79198",
        "customer_role": "both",
        "sections": [
            ("1. Tổng quan các Phương thức Thanh toán", 
             "Shopee hỗ trợ đa dạng các phương thức thanh toán an toàn cho cả Người mua và Người bán: "
             "1. Ví ShopeePay: Thanh toán trực tuyến tức thì, miễn phí giao dịch và nhận nhiều ưu đãi giảm giá. "
             "2. Thanh toán khi nhận hàng (COD): Người mua thanh toán tiền mặt trực tiếp cho nhân viên giao hàng khi nhận sản phẩm. "
             "3. Thẻ Tín dụng / Thẻ Ghi nợ: Hỗ trợ các thẻ quốc tế Visa, Mastercard, JCB, American Express và thẻ ATM nội địa Napas. "
             "4. Trả góp 0% qua Thẻ Tín dụng: Áp dụng cho các đơn hàng hợp lệ có giá trị tối thiểu từ 3.000.000 VNĐ. "
             "5. Dịch vụ SPayLater: Mua trước trả sau với hạn mức lên đến 10.000.000 VNĐ, linh hoạt thanh toán trong 1, 2, 3, 6 hoặc 12 tháng. "
             "6. Chuyển khoản ngân hàng qua mã VietQR / Mobile Banking: Thanh toán nhanh chóng qua QR Code. "
             "7. Apple Pay & Google Pay: Thanh toán một chạm an toàn trên các thiết bị di động hỗ trợ."),
            ("2. Hướng dẫn thay đổi Phương thức Thanh toán", 
             "Người mua chỉ có thể thay đổi phương thức thanh toán đối với đơn hàng đang ở trạng thái 'Chờ thanh toán'. Đối với đơn hàng đã chuyển sang trạng thái 'Đang xử lý' hoặc 'Đang giao', phương thức thanh toán không thể thay đổi."),
            ("3. Bảo mật giao dịch thanh toán", 
             "Tất cả thông tin thanh toán trên Shopee được bảo mật theo tiêu chuẩn PCI-DSS Level 1. Hệ thống đảm bảo tiền của Người mua được Shopee giữ an toàn (Shopee Đảm Bảo) và chỉ thanh toán cho Người bán khi Người mua xác nhận nhận hàng thành công.")
        ]
    },
    {
        "filename": "privacy-policy-shopee.pdf",
        "title": "CHÍNH SÁCH BẢO MẬT THÔNG TIN VÀ DỮ LIỆU CÁ NHÂN SHOPEE",
        "url": "https://help.shopee.vn/portal/4/article/77244",
        "customer_role": "both",
        "sections": [
            ("1. Thu thập dữ liệu cá nhân", 
             "Shopee thu thập thông tin cá nhân của người dùng khi đăng ký tài khoản, thực hiện giao dịch hoặc tương tác trên nền tảng. Dữ liệu bao gồm: Họ và tên, số điện thoại, địa chỉ nhận hàng, địa chỉ email, thông tin tài khoản ngân hàng, vị trí địa lý và lịch sử duyệt web/mua hàng."),
            ("2. Mục đích sử dụng thông tin", 
             "Thông tin cá nhân được sử dụng để: Xử lý và vận chuyển đơn hàng; xác minh danh tính tài khoản; cung cấp dịch vụ hỗ trợ khách hàng; gửi thông báo cập nhật đơn hàng và chương trình khuyến mãi; phát hiện và ngăn chặn các hành vi gian lận, truy cập trái phép."),
            ("3. Chia sẻ thông tin với bên thứ ba", 
             "Shopee cam kết không bán hay tự ý chia sẻ dữ liệu cá nhân cho bên thứ ba vì mục đích thương mại. Thông tin chỉ được chia sẻ cho: Các đơn vị vận chuyển đối tác (Shopee Express, GHN, GHTK, Viettel Post) để giao hàng; các đối tác cổng thanh toán để xử lý giao dịch; và cơ quan nhà nước có thẩm quyền khi có yêu cầu hợp pháp."),
            ("4. Quyền riêng tư của người dùng", 
             "Người dùng có quyền xem, cập nhật hoặc yêu cầu xóa dữ liệu cá nhân bất kỳ lúc nào thông qua phần Cài đặt Tài khoản trên ứng dụng Shopee hoặc gửi yêu cầu tới Bộ phận Bảo vệ Dữ liệu của Shopee.")
        ]
    },
    {
        "filename": "seller-listing-regulations-shopee.pdf",
        "title": "QUY ĐỊNH ĐĂNG BÁN SẢN PHẨM DÀNH CHO NGƯỜI BÁN SHOPEE",
        "url": "https://help.shopee.vn/portal/4/article/77245",
        "customer_role": "seller",
        "sections": [
            ("1. Danh mục sản phẩm bị cấm đăng bán", 
             "Người bán không được phép đăng bán: Hàng giả, hàng nhái, sản phẩm vi phạm quyền sở hữu trí tuệ; vũ khí, chất cháy nổ, công cụ hỗ trợ; chất ma túy, thuốc lá điện tử; thực phẩm tươi sống không đảm bảo an toàn vệ sinh; động vật hoang dã; sản phẩm văn hóa phẩm độc hại."),
            ("2. Quy chuẩn tiêu đề và hình ảnh sản phẩm", 
             "Tiêu đề sản phẩm phải mô tả chính xác mặt hàng, không chứa từ khóa rác hoặc thương hiệu không liên quan. Hình ảnh sản phẩm phải rõ nét, chính chủ hoặc được cấp quyền, không chứa mờ nhòe hay chèn logo của các sàn thương mại điện tử đối thủ."),
            ("3. Xử lý vi phạm quy định đăng bán", 
             "Sản phẩm vi phạm sẽ bị tạm khóa hoặc xóa khỏi sàn. Người bán vi phạm sẽ bị cộng điểm phạt Sao Quả Tạ, giới hạn số lượng sản phẩm đăng bán, hoặc bị đóng tài khoản Người bán vĩnh viễn đối với các vi phạm nghiêm trọng.")
        ]
    }
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc da san sang: {DATA_DIR}")


def build_pdf_file(policy: dict):
    """
    Sử dụng FPDF2 để tạo file PDF có hỗ trợ tiếng Việt Unicode (dùng Arial font).
    """
    pdf = FPDF()
    pdf.add_page()

    # Tìm font Arial hoặc Segoe UI trên Windows để hỗ trợ Unicode tiếng Việt
    font_path = r"C:\Windows\Fonts\arial.ttf"
    font_bold_path = r"C:\Windows\Fonts\arialbd.ttf"

    if os.path.exists(font_path) and os.path.exists(font_bold_path):
        pdf.add_font("ArialVN", "", font_path)
        pdf.add_font("ArialVN", "B", font_bold_path)
        font_name = "ArialVN"
    else:
        font_name = "Helvetica"  # Fallback

    # Header / Title
    pdf.set_font(font_name, "B", 16)
    pdf.multi_cell(0, 10, policy["title"], new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Metadata
    pdf.set_font(font_name, "", 10)
    pdf.cell(0, 6, f"Nguon: {policy['url']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Customer Role: {policy['customer_role'].upper()}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Content Sections
    for sec_title, sec_body in policy["sections"]:
        pdf.set_font(font_name, "B", 12)
        pdf.multi_cell(0, 8, sec_title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 11)
        pdf.multi_cell(0, 6, sec_body, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    filepath = DATA_DIR / policy["filename"]
    pdf.output(str(filepath))
    size_kb = filepath.stat().st_size / 1024
    print(f"  [OK] Da tao PDF: {policy['filename']} ({size_kb:.1f} KB) - Role: {policy['customer_role']}")


def collect_all_legal_docs():
    """Khởi tạo toàn bộ văn bản chính sách PDF trong data/landing/legal/."""
    setup_directory()
    print("\nDang tao cac file PDF chinh sach e-commerce (Shopee Vietnam)...")
    for policy in POLICIES:
        build_pdf_file(policy)
    print(f"\n[OK] Hoan tat Task 1! Da tao {len(POLICIES)} file PDF trong {DATA_DIR}")


if __name__ == "__main__":
    collect_all_legal_docs()
