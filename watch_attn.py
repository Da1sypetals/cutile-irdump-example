# [watch]
# tensor((8, 16, 1024, 64), dtype="float32")
# tensor((8, 16, 1024, 64), dtype="float32")
# tensor((8, 16, 1024, 64), dtype="float32")
# tensor((8, 16, 1024, 64), dtype="float32")
# 64
# 32
# 32

import cuda.tile as ct


@ct.kernel
def flash_attention_forward_v2(
    q,
    k,
    v,
    out,
    hidden_size: ct.Constant,
    br: ct.Constant,
    bc: ct.Constant,
):
    ib = ct.bid(0)
    ih = ct.bid(1)
    Tc = k.shape[0] // bc
    qi = ct.load(q, index=(ib, ih, ct.bid(2), 0), shape=(1, 1, br, hidden_size))

    qi = ct.reshape(qi, (br, hidden_size))

    oi = ct.full((br, hidden_size), 0.0, dtype=q.dtype)
    li = ct.full((br, 1), 0.0, dtype=q.dtype)
    mi = ct.full((br, 1), -1e10, dtype=q.dtype)

    for j in range(0, Tc):
        # kj = ct.load(k, index=(0, j), shape=(hidden_size, bc), order="F")
        # vj = ct.load(v, index=(j, 0), shape=(bc, hidden_size))
        kj = ct.load(k, index=(ib, ih, 0, j), shape=(1, 1, hidden_size, bc), order="F")
        vj = ct.load(v, index=(ib, ih, j, 0), shape=(1, 1, bc, hidden_size))
        kj = ct.reshape(kj, (hidden_size, bc))
        vj = ct.reshape(vj, (bc, hidden_size))
        sij = ct.matmul(qi, kj) / hidden_size**0.5
        mij = ct.max(sij, axis=-1, keepdims=True)
        mi_mij = ct.cat((mi, mij), axis=-1)
        mi_new = ct.max(mi_mij, axis=-1, keepdims=True)
        pij = ct.exp(sij - mi_new)
        lij = ct.sum(pij, axis=-1, keepdims=True)
        exp_mi = ct.exp(mi - mi_new)
        li_new = li * exp_mi + lij
        oi = ct.mma(pij, vj, oi * exp_mi)
        li = li_new
        mi = mi_new

    oi = oi / li
    ct.store(out, index=(ib, ih, ct.bid(2), 0), tile=oi)
