def smooth_loss(l, f, i, r):
    return (l + r * min(f, i))/(min(f, i+1))
                