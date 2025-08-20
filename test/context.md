[page 22] _20_ _Chương 1. Các ứng dụng của phép tính vi phân trong hình học_


d. d −→
� p ( t ) ∧−→ q ( t ) �
dt



,
�����



p 3 ( t ) p 1 ( t )
����� q 3 ( t ) q 1 ( t )



,
�����



p 1 ( t ) p 2 ( t )
����� q 1 ( t ) q 2 ( t )



�����



= [d]

dt


= ... p 2 ( t ) p 3 ( t )
������ q 2 ( t ) q 3 ( t )



�



p 3 ( t ) p 1 [′] [(] [t] [)]

�����, ����� q 3 ( t ) q 1 [′] [(] [t] [)]


p 3 [′] [(] [t] [)] p 1 ( t )

�����, ����� q 3 [′] [(] [t] [)] q 1 ( t )



p 1 ( t ) p 2 [′] [(] [t] [)]

�����, ����� q 1 ( t ) q 2 [′] [(] [t] [)]


p 1 [′] [(] [t] [)] p 2 ( t )

�����, ����� q 1 [′] [(] [t] [)] q 2 ( t )



������

������



=


+



p 2 ( t ) p 3 [′] [(] [t] [)]
������ q 2 ( t ) q 3 [′] [(] [t] [)]


p 2 [′] [(] [t] [)] p 3 ( t )
������ q 2 [′] [(] [t] [)] q 3 ( t )




[(] [t] [)] + [d] [−→] [p] [(] [t] [)]

dt dt




[(] [t] [)]
= [−→] p ( t ) ∧ [d] [−→] [q]



∧ [−→] q ( t )
dt



**Bài tập 1.5.** Viết phương trình tiếp tuyến và pháp diện của đường:



a. b.

[page 30] _28_ _Chương 2. Tích phân bội_


**Chú ý 2.3.** Nếu tồn tại tích phân kép f ( x, y ) dxdy thì ta nói hàm số f ( x, y ) khảtích
��

D

trong miền D .


**Tính chất cơ bản:**


  Tính chất tuyến tính:



��



f ( x, y ) dxdy +
��
D D




[ f ( x, y ) + g ( x, y )] dxdy =
��
D D



g ( x, y ) dxdy

D



��



k f ( x, y ) dxdy = k
��
D D



f ( x, y ) dxdy

D




- Tính chất cộng tính: Nếu D = D 1 ∪ D 2, ởđó D 1 và D 2 không "chồng" lên nhau (có thể
ngoại trừphần biên) thì



��



f ( x, y ) dxdy =
��
D D



f ( x, y ) dxdy.



D 1



f ( x, y ) dxdy +
��

D 2










#### **1.2 Tính tích phân kép trong hệtoạđộDescartes**

Đểtính các tích phân hai lớp, ta cần phải đưa vềtính các tích phân lặp.


1. Nếu D là miền hình chữnhật ( D ) : a ⩽ x ⩽ b, c ⩽ y ⩽ d thì ta có thểsửdụng một
trong hai tích phân lặp



d


dy

�


c



d


f ( x, y ) dx.

�


c



d


f ( x, y ) dy =

�


c


28



f ( x, y ) dxdy =

��

D



b


dx

�


a

[page 25] # **CHƯƠNG 2**

## **T ÍCH PHÂN BỘI**

#### § 1. T ÍCH PHÂN KÉP **1.1 Định nghĩa**

**Diện tích và tích phân xác định**


Cho f ( x ) là một hàm sốxác định với a ≤ x ≤ b. Đầu tiên ta chia khoảng [ a, b ] này thành
n khoảng nhỏ [ x i − 1, x i ] với độdài bằng nhau ∆x = [b] [−] n [a] và chọn trong mỗi khoảng đó một

điểm x i [∗] [bất kì. Sau đó lập tổng Riemann]


n
#### S ( n ) = ∑ f ( x i [∗] [)] [∆][x]

i = 1


23