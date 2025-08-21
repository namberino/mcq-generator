[page 19] ### **4.4** **Các lỗhổng được khai thác**

#### **4.4.1 Command injection trong tính năng chuẩn đoán** **ping**

Lỗhổng này có mã CVE là **CVE-2024-51186** [13]. Đây là lỗhổng trong
dịch vụ **ncc2** của thiết bịnày. Trong dịch vụ **ncc2** này, có một endpoint xử
lý CGI request là **ping.ccp** . Endpoint này cho phép chuẩn đoán các thiết
bịqua mạng bằng cách "ping"các thiết bịđó. Dưới đây là ảnh giao diện cho
phép người dùng tương tác với tính năng này.


18

[page 20] Dưới đây là một request tới **ping.ccp** đã được bắt trong Burp Suite:


Sau khi dịch ngược file **ncc2**, chúng ta có thểthấy được là hàm
**FUN_0049e128** sẽxửlý các POST request tới **ping.ccp** . Hàm này
sẽlấy địa chỉping từtham số **ping_addr** trong request.


19

[page 23] Hàm **callback_ccp_ddns_check** có sửdụng hàm **doCheck** (được sử
dụng khi tham số **ccp_act** trong request tới **ddns_check.ccp** được set
thành **doCheck** ) đểthu thập thông tin DDNS: Tên host, tên người dùng,
mật khẩu.


Những tham sốnày sẽđược đưa vào hàm **system** . Hàm này sẽchạy
**ddns_check.c** đểxửlý request.


Hàm **doCheck** này không có bất kì chức năng nào đểkiểm tra xem đầu vào
có bịcommand injection hay không. Dưới đây là minh họa một payload mở


22