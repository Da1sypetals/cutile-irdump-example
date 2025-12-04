import cuda.tile as ct
import torch


@ct.kernel
def softmax(x, out, r):
    c = ct.load(x, (0, ct.bid(0)), (x.shape[0], r))
    max = ct.max(c, axis=0, keepdims=True)
    num = ct.exp(c - max)
    den = ct.sum(num, axis=0, keepdims=True)
    smax = num / den
    ct.store(out, (0, ct.bid(0)), smax)


x = torch.randn(512, 128, device="cuda")
out = torch.empty(512, 128, device="cuda")
ct.launch(0, (64,), softmax, x, out, 8)

print(out.shape)
