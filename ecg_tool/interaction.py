"""Pointer-centered wheel zoom for source and CSV views."""
def zoom_limits(limits, anchor, steps):
    factor = .85 ** max(-10, min(10, steps))
    return anchor-(anchor-limits[0])*factor, anchor+(limits[1]-anchor)*factor
