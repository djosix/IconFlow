'''
Copyright (c) 2022 Yuankui Lee
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import wx
from typing import Dict

from .template import TemplateFrame
from .event import EventManager
from .impl_sketch_canvas import SketchCanvasImplMixin
from .impl_style_palette import StylePaletteImplMixin
from .impl_display_grid import DisplayGridImplMixin
from .impl_model_inference import ModelInferenceImplMixin


class AppFrame(
    ModelInferenceImplMixin,
    DisplayGridImplMixin,
    StylePaletteImplMixin,
    SketchCanvasImplMixin,
    TemplateFrame,
    EventManager
):
    def __init__(self, parent):
        # Create event channels
        EventManager.__init__(self)

        # Provide for wxFormBuilder codegen
        self.previewSlots: Dict[wx.StaticBitmap] = {}
        TemplateFrame.__init__(self, parent)

        SketchCanvasImplMixin.__init__(self)
        StylePaletteImplMixin.__init__(self)
        DisplayGridImplMixin.__init__(self)
        ModelInferenceImplMixin.__init__(self)


class App(wx.App):
    def OnInit(self):
        self.frame = AppFrame(None)
        self.frame.Show()
        return True
