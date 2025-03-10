print("init")
import torch, time

print("after importing")
gpu0 = torch.device("cuda:0")
gpu1 = torch.device("cuda:1")
gpu2 = torch.device("cuda:2")
gpu3 = torch.device("cuda:3")
scale = 50000
matrix0 = torch.randn(scale, scale, device=gpu0)
matrix1 = torch.randn(scale, scale, device=gpu1)
matrix2 = torch.randn(scale, scale, device=gpu2)
matrix3 = torch.randn(scale, scale, device=gpu3)
print("after num init")

cnt = 0
while True:
    cnt += 1
    matrix0 @ matrix0
    matrix1 @ matrix1
    matrix2 @ matrix2
    matrix3 @ matrix3
    print(cnt)
    time.sleep(5)
