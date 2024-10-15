import mlx.core as mx

a = mx.ones([4, 2])
b = mx.array([[10, 10, 10, 10], [20, 20, 20, 20]])

print(a @ b)


def by_definition(a, b):
    c = mx.zeros([a.shape[0], b.shape[1]])
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            c[i, j] = mx.inner(a[i, :], b[:, j])
    return c


print(by_definition(a, b))


def tile_1x1(a, b):
    c = mx.zeros([a.shape[0], b.shape[1]])
    for i in range(a.shape[0]):
        for k in range(a.shape[1]):
            for j in range(b.shape[1]):
                c[i, j] += a[i, k] * b[k, j]
    return c


print(tile_1x1(a, b))


def tile_txt(a, b, t=2):
    c = mx.zeros([a.shape[0], b.shape[1]])
    for i in range(0, a.shape[0], t):
        for k in range(0, a.shape[1], t):
            for j in range(0, b.shape[1], t):
                at = a[i : i + t, k : k + t]
                bt = b[k : k + t, j : j + t]
                c[i : i + t, j : j + t] += at @ bt
    return c


print(tile_txt(a, b))


def tile_1xt(a, b, t=2):
    c = mx.zeros([a.shape[0], b.shape[1]])
    for i in range(0, a.shape[0], t):
        for j in range(0, b.shape[1], t):
            at = a[i : i + t, :]
            bt = b[:, j : j + t]
            c[i : i + t, j : j + t] += at @ bt
    return c


print(tile_1xt(a, b))
