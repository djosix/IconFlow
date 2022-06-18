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
import traceback
from PIL import Image

from .resource import Paths
from .event import EventManager
from .utils import bitmapToImage, imageToBitmap


class SketchCanvasImplMixin(EventManager):
    # TemplateFrame
    sketchCanvas: wx.StaticBitmap

    newButton: wx.Button
    openButton: wx.Button
    saveButton: wx.Button
    pencilButton: wx.Button
    eraserButton: wx.Button
    undoButton: wx.Button
    redoButton: wx.Button

    def __init__(self):
        
        self.sketchCanvas.Bind(wx.EVT_IDLE, self.sketchCanvasOnIdle)

        self.sketchSize = tuple(self.sketchCanvas.GetSize())
        self.__defaultImage = Image.new('RGB', self.sketchSize, (255, 255, 255))
        assert self.sketchSize == (512, 512), repr(self.sketchSize)
        self.__currentBitmap: wx.Bitmap = wx.Bitmap.FromBuffer(*self.sketchSize, self.__defaultImage.tobytes())
        self.sketchCanvas.SetBitmap(self.__currentBitmap)

        self.__lastPoint = None # last down or drag point
        self.__shouldRefresh = False # True if sketch canvas should re-paint
        self.__inEraseMode = False # erase mode enabled

        self.__drawingPen = wx.Pen(wx.BLACK, 4)
        self.__erasingPen = wx.Pen(wx.WHITE, 18)

        drawingCursorImage = wx.Image(Paths.DRAWING_CURSOR, wx.BITMAP_TYPE_PNG)
        drawingCursorImage.SetOption(wx.IMAGE_OPTION_CUR_HOTSPOT_X, 0)
        drawingCursorImage.SetOption(wx.IMAGE_OPTION_CUR_HOTSPOT_Y, 30)
        self.__drawingCursor = wx.Cursor(drawingCursorImage)
        self.sketchCanvas.SetCursor(self.__drawingCursor)

        erasingCursorImage = wx.Image(Paths.ERASING_CURSOR, wx.BITMAP_TYPE_PNG)
        erasingCursorImage.SetOption(wx.IMAGE_OPTION_CUR_HOTSPOT_X, 4)
        erasingCursorImage.SetOption(wx.IMAGE_OPTION_CUR_HOTSPOT_Y, 24)
        self.__erasingCursor = wx.Cursor(erasingCursorImage)

        self.__isDrawing = False

        self.__previewBitmap = None
        self.subscribe('preview', self.__updatePreviewImage)

        self.__imageHistory = []
        self.__imageHistoryIndex = 0

        self.Bind(wx.EVT_KEY_DOWN, self.onKeyDown)
        self.SetFocus()

    #==============================================================================
    # Private Methods
    #==============================================================================

    def __drawLines(self, points):

        dc = wx.BufferedDC(None, self.__currentBitmap)
        dc.SetPen([self.__drawingPen, self.__erasingPen][self.__inEraseMode])
        dc.DrawLines(points)

        self.__shouldRefresh = True
        
    def __updatePreviewImage(self, image):

        self.__shouldRefresh = bool(self.__previewBitmap or image)
        self.__previewBitmap = None

        if image is not None:
            image = image.resize(self.__currentBitmap.Size, Image.NEAREST)
            self.__previewBitmap = imageToBitmap(image)

        if self.__shouldRefresh:
            wx.CallAfter(self.sketchCanvas.Refresh, False)
        
    def __notifySketch(self):

        self.notify('sketch', bitmapToImage(self.__currentBitmap))
    
    #==============================================================================
    # Canvas Events
    #==============================================================================

    def sketchCanvasOnLeftDown(self, event: wx.MouseEvent):

        self.sketchCanvas.CaptureMouse()

        self.__lastPoint = event.GetPosition()
        self.__drawLines([self.__lastPoint] * 2)
        self.__isDrawing = True

        self.SetFocus()

    def sketchCanvasOnLeftUp(self, event: wx.MouseEvent):

        if self.sketchCanvas.HasCapture():
            self.sketchCanvas.ReleaseMouse()
            self.__notifySketch()

            self.__imageHistory = self.__imageHistory[:self.__imageHistoryIndex]
            self.__imageHistory.append(bitmapToImage(self.__currentBitmap))
            self.__imageHistoryIndex += 1

        self.__isDrawing = False

    def sketchCanvasOnLeaveWindow(self, event: wx.MouseEvent):

        self.sketchCanvasOnLeftUp(event)
    
    def sketchCanvasOnEnterWindow(self, event: wx.MouseEvent):

        self.__updatePreviewImage(None)

    def sketchCanvasOnMotion(self, event: wx.MouseEvent):

        point = event.GetPosition()

        if event.Dragging() and event.LeftIsDown() and self.__isDrawing:
            if self.__lastPoint != point:
                self.__drawLines([self.__lastPoint, point])
        
        self.__lastPoint = point

        if self.__isDrawing:
            self.__notifySketch()

    def sketchCanvasOnPaint(self, event: wx.PaintEvent):

        bitmap = self.__previewBitmap or self.__currentBitmap

        try:
            wx.BufferedPaintDC(self.sketchCanvas, bitmap)
        except RuntimeError:
            pass
    
    def sketchCanvasOnIdle(self, event: wx.IdleEvent):

        if self.__shouldRefresh:
            self.sketchCanvas.Refresh(False)
            self.__shouldRefresh = False

        if self.__inEraseMode:
            self.sketchCanvas.SetCursor(self.__erasingCursor)
            self.pencilButton.Enable()
            self.eraserButton.Disable()
        else:
            self.sketchCanvas.SetCursor(self.__drawingCursor)
            self.pencilButton.Disable()
            self.eraserButton.Enable()
        
        if self.__imageHistoryIndex < len(self.__imageHistory):
            self.redoButton.Enable()
        else:
            self.redoButton.Disable()
        
        if self.__imageHistoryIndex > 0:
            self.undoButton.Enable()
        else:
            self.undoButton.Disable()

    #==============================================================================
    # Button Events
    #==============================================================================

    def newButtonOnButtonClick(self, event: wx.CommandEvent):
        
        self.__currentBitmap = imageToBitmap(self.__defaultImage)
        self.__imageHistory.clear()
        self.__imageHistoryIndex = 0
        self.sketchCanvas.Refresh(False)

        self.__notifySketch()

    def openButtonOnButtonClick(self, event: wx.CommandEvent):
        
        with wx.FileDialog(
            self, 'Open PNG file',
            wildcard='PNG files (*.png)|*.png',
            style=(wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        ) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            try:
                image = Image.open(fileDialog.GetPath())
                image = image.convert('RGB')
                image = image.resize(self.sketchSize, Image.NEAREST)

            except:
                wx.LogError(traceback.format_exc())
                return
            
            self.__currentBitmap = imageToBitmap(image)
            self.sketchCanvas.Refresh(False)
            self.__notifySketch()

    def saveButtonOnButtonClick(self, event: wx.CommandEvent):
        
        if self.__previewBitmap is None:
            bitmap = self.__currentBitmap
            default = 'sketch.png'
            message = 'Save sketch'
        else:
            bitmap = self.__previewBitmap
            default = 'icon.png'
            message = 'Save icon'
        
        with wx.FileDialog(
            self,
            message=message,
            defaultFile=default,
            wildcard='PNG files (*.png)|*.png',
            style=(wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        ) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            try:
                image = bitmapToImage(bitmap)
                image = image.resize((512, 512), Image.BICUBIC)
                image.save(fileDialog.GetPath())

            except:
                wx.LogError(traceback.format_exc())

    def pencilButtonOnButtonClick(self, event: wx.CommandEvent):
        
        self.__inEraseMode = False

    def eraserButtonOnButtonClick(self, event: wx.CommandEvent):
        
        self.__inEraseMode = True

    def undoButtonOnButtonClick(self, event: wx.CommandEvent):
        
        self.__imageHistoryIndex = max(0, self.__imageHistoryIndex - 1)

        if self.__imageHistoryIndex > 0:
            currentImage = self.__imageHistory[self.__imageHistoryIndex - 1]
        else:
            currentImage = self.__defaultImage
        
        self.__currentBitmap = imageToBitmap(currentImage)
        self.__shouldRefresh = True
        self.__notifySketch()

    def redoButtonOnButtonClick(self, event: wx.CommandEvent):
        
        if self.__imageHistoryIndex >= len(self.__imageHistory):
            return
        

        if self.__imageHistory:
            currentImage = self.__imageHistory[self.__imageHistoryIndex]
        else:
            currentImage = self.__defaultImage

        self.__imageHistoryIndex += 1
        
        self.__currentBitmap = imageToBitmap(currentImage)
        self.__shouldRefresh = True
        self.__notifySketch()
    
    def onKeyDown(self, event: wx.KeyEvent):
        if event.GetKeyCode() == ord('Z'):
            if event.ControlDown():
                if event.ShiftDown():
                    self.redoButtonOnButtonClick(None)
                else:
                    self.undoButtonOnButtonClick(None)
