import torch
# x = torch.arange(12, dtype=torch.float32)
# print("x:",x)
# print("x.numel",x.numel())

# x_re = x.reshape(3,4)
# print(x_re)

# print(x_re[1,3])
# print(x_re[1:3])
# print(x_re[:,1:3])
# print(x_re[-1,:])

# x_re[1,2] = 100
# print(x_re)

# x_re[:2,:] = 100
# print(x_re)
# all_zero = torch.zeros(2,3,4)
# all_one = torch.ones(3,4,2)

# # print(all_zero)
# # print(all_one)

# gaussian = torch.randn(3,4)
# print(gaussian)

# print(torch.exp(x))

# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])
# print("+:", x + y)
# print("-:", x - y)
# print("*:", x * y)
# print("/:", x / y)
# print("**:", x ** y)

# x = torch.arange(12, dtype=torch.float32).reshape(3,4)
# y = torch.tensor([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
# # print(torch.cat((x,y), dim = 0))
# # print(torch.cat((x,y), dim = 1))


# print(x.sum())

a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))

print(a)
print(b)
print(a+b)