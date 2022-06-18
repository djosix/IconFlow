import wx
import time
import numpy as np
import threading
import queue
from PIL import Image

class DebouncedCaller:
    def __init__(self, callback, delay):
        self.callback = callback
        self.delay = delay
        self.lastTime = time.time()

    def __call__(self, *args, **kwargs):
        currentTime = time.time()

        if currentTime - self.lastTime > self.delay:
            self.callback(*args, **kwargs)
            self.lastTime = currentTime

def bitmapToImage(bitmap: wx.Bitmap) -> Image.Image:
    wxImage = bitmap.ConvertToImage()
    buffer = np.frombuffer(wxImage.GetDataBuffer(), dtype=np.uint8)
    image = Image.frombuffer('RGB', tuple(wxImage.GetSize()), buffer)
    return image

def imageToBitmap(image: Image.Image) -> wx.Bitmap:
    bitmap = wx.Bitmap.FromBuffer(*image.size, image.convert('RGB').tobytes())
    return bitmap

def spawn(target, *args, **kwargs):
    thread = threading.Thread(
        target=target,
        args=args,
        kwargs=kwargs,
        daemon=True
    )
    thread.start()
    return thread

def queue_clear(q: queue.Queue):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return

def queue_put(q: queue.Queue, value):
    try:
        q.put_nowait(value)
        return True
    except queue.Full:
        return False

def queue_get(q: queue.Queue, default=None):
    try:
        return q.get_nowait()
    except queue.Empty:
        return default
