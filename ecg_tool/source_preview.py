"""Read-only source preview: fixed red zero references, wheel zoom, drag pan."""
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .extractor import _load_first_page, _orient_page, _second_internal_major_grid_y


from .interaction import zoom_limits

def preview_source(parent, pdf_path, settings):
    image, dpi, _ = _load_first_page(pdf_path, settings.render_dpi)
    image, _, bands, _ = _orient_page(image)
    zeros = [_second_internal_major_grid_y(image, band, dpi/25.4*settings.gain_mm_per_mV) for band in bands]
    window = tk.Toplevel(parent)
    window.title("ECG Source Preview - Fixed Default Zero Lines")
    window.geometry("1180x800")
    window.transient(parent)
    ttk.Label(window, text="Red = default 0 mV (second internal major line from bottom; bottom border excluded). Hover and scroll to zoom. Drag to pan. Double-click to reset.", wraplength=1120).pack(pady=10)
    figure = Figure(figsize=(11, 6.5), dpi=100)
    axes = figure.subplots(3, 1)
    original = {}
    for i, (ax, band, zero) in enumerate(zip(axes, bands, zeros)):
        y0, y1, x0, x1 = band
        ax.imshow(image[y0:y1, x0:x1], extent=(x0-.5,x1-.5,y1-.5,y0-.5), aspect="auto")
        ax.axhline(zero, color="#e5232e", linewidth=1.2)
        ax.set_title(f"Segment {i+1} | default zero: image y = {zero:.2f} px", fontsize=10)
        ax.set_xticks([])
        ax.set_ylabel("Image y (px)")
        original[ax] = (ax.get_xlim(), ax.get_ylim())
    figure.tight_layout()
    canvas = FigureCanvasTkAgg(figure, master=window)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    drag = None

    def reset():
        for ax, (xl,yl) in original.items():
            ax.set_xlim(xl)
            ax.set_ylim(yl)
        canvas.draw_idle()

    def scroll(event):
        if event.inaxes not in original or event.xdata is None or event.ydata is None:
            return
        ax=event.inaxes
        ax.set_xlim(zoom_limits(ax.get_xlim(), event.xdata, event.step))
        ax.set_ylim(zoom_limits(ax.get_ylim(), event.ydata, event.step))
        canvas.draw_idle()

    def press(event):
        nonlocal drag
        if event.inaxes not in original:
            return
        if event.dblclick:
            reset()
            return
        if event.button == 1:
            ax=event.inaxes
            drag=(ax,event.x,event.y,ax.get_xlim(),ax.get_ylim())

    def motion(event):
        if drag is None:
            return
        ax,x,y,xl,yl=drag
        if event.inaxes is not ax or ax.bbox.width<=0 or ax.bbox.height<=0:
            return
        dx=(event.x-x)/ax.bbox.width*(xl[1]-xl[0])
        dy=(event.y-y)/ax.bbox.height*(yl[1]-yl[0])
        ax.set_xlim(xl[0]-dx,xl[1]-dx)
        ax.set_ylim(yl[0]-dy,yl[1]-dy)
        canvas.draw_idle()

    def release(event):
        nonlocal drag
        drag=None

    for name, callback in (("scroll_event",scroll),("button_press_event",press),("motion_notify_event",motion),("button_release_event",release)):
        canvas.mpl_connect(name, callback)
    buttons=ttk.Frame(window)
    buttons.pack(pady=8)
    ttk.Button(buttons,text="Reset View",command=reset).pack(side="left",padx=8)
    ttk.Button(buttons,text="Close",command=window.destroy).pack(side="left",padx=8)
    # No selection or confirmation is required. The preview cannot change references.
