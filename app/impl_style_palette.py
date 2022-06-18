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
import pickle
import numpy as np

from .event import EventManager
from .resource import Paths
from .utils import DebouncedCaller

class StylePaletteImplMixin(EventManager):
    # TemplateFrame
    stylePalette: wx.StaticBitmap

    class Config:
        def __init__(self, size):
            self.width, self.height = size

            self.horizontalCenter = self.width // 2
            self.verticalCenter = self.height // 2

            self.scaleWidth, self.scaleHeight = 480, 480

            self.styleWidth, self.styleHeight = 42, 16

            self.horizontalTick = self.scaleWidth // 6
            self.verticalTick = self.scaleHeight // 6
            
            self.padding = 10

            self.gridLeft = self.horizontalCenter - self.scaleWidth // 2
            self.gridRight = self.horizontalCenter + self.scaleWidth // 2

            self.gridTop = self.verticalCenter - self.scaleHeight // 2
            self.gridBottom = self.verticalCenter + self.scaleHeight // 2

            self.axisColor = wx.Colour(0, 0, 0)
            self.axisLineWidth = 2
            self.axisLabelTop = 'soft'
            self.axisLabelBottom = 'hard'
            self.axisLabelLeft = 'warm'
            self.axisLabelRight = 'cool'

            self.gridColor = wx.Colour(127, 127, 127)
            self.gridLineWidth = 1

            self.styleLineColor = wx.Colour(127, 127, 127)
            self.styleLineWidth = 1

            self.styleHoverLineColor = wx.Colour(96, 96, 96)
            self.styleHoverLineWidth = 3

            self.styleBgColor = wx.WHITE
            self.styleFgColor = self.styleLineColor
            self.styleFont = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT)
            self.styleHoverBgColor = wx.WHITE
            self.styleHoverFgColor = self.styleHoverLineColor
            self.styleHoverFont = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT)

            self.styleLocFont = wx.Font(wx.FONTSIZE_SMALL, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_HEAVY)
            self.styleLocBgColor = wx.WHITE
            self.styleLocFgColor = wx.BLACK

            self.markSize = 16
            self.markPen = wx.Pen(wx.Colour(240, 40, 120), 3)
            self.markFont = wx.Font(wx.FONTSIZE_SMALL, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_EXTRAHEAVY)
            self.markFgColor = wx.RED
            self.markBgColor = wx.WHITE

            self.backgroundColor = wx.WHITE

        def styleLocToCanvasPos(self, styleLoc):
            return (
                styleLoc[0] * self.horizontalTick + self.horizontalCenter,
                -styleLoc[1] * self.verticalTick + self.verticalCenter
            )
        
        def canvasPosToStyleLoc(self, canvasPos):
            return (
                (canvasPos[0] - self.horizontalCenter) / self.horizontalTick,
                -(canvasPos[1] - self.verticalCenter) / self.verticalTick
            )
        
        def isStyleLocInRange(self, styleLoc):
            return (-3 <= styleLoc[0] <= 3) and (-3 <= styleLoc[1] <= 3)
        
        def isCanvasPosInRange(self, canvasPos):
            return self.isStyleLocInRange(self.canvasPosToStyleLoc(canvasPos))

    def __init__(self):

        self.stylePalette.Bind(wx.EVT_IDLE, self.stylePaletteOnIdle)

        self.__shouldRefresh = True

        self.__currentMousePos = None
        self.__selectedCanvasPos = None
        self.__sampleTemperature = 0.4

        self.__config = self.Config(self.stylePalette.Size)
        
        with open(Paths.COLOR_IMAGE_SCALE, 'rb') as f:
            self.__styleGroups = pickle.load(f)

            self.__styleCombs = [
                (group, name, location, comb)
                for group, combs in self.__styleGroups.items()
                for name, location, comb in combs
            ]

            self.__stylePoss = np.array([
                self.__config.styleLocToCanvasPos(location)
                for _, combs in self.__styleGroups.items()
                for _, location, _ in combs
            ])
        
        self.__hoveredStyleIndex = None
        self.__selectedStyleIndex = None

        self.__notifyLocationDebounced = DebouncedCaller(self.__notifyLocation, 0.2)
    
    def __getStyleIndexByCanvasPos(self, canvasPos):

        posX, posY = canvasPos
        styleXs = self.__stylePoss[:, 0]
        styleYs = self.__stylePoss[:, 1]

        left = styleXs - self.__config.styleWidth / 2
        right = styleXs + self.__config.styleWidth / 2
        top = styleYs - self.__config.styleHeight / 2
        bottom = styleYs + self.__config.styleHeight / 2

        where = np.where((
            (left <= posX) & (posX <= right) &
            (top <= posY) & (posY <= bottom)
        ))[0]

        return where[0] if where.size else None

    def stylePaletteOnLeftDown(self, event: wx.MouseEvent):

        mousePos = event.GetPosition()
        self.__currentMousePos = mousePos

        self.stylePalette.CaptureMouse()

        if self.__config.isCanvasPosInRange(mousePos):
            self.__selectedCanvasPos = mousePos
            self.__selectedStyleIndex = self.__getStyleIndexByCanvasPos(mousePos)

            self.__notifySample()
            wx.CallLater(1, self.__notifyLocationDebounced)

            self.__shouldRefresh = True
    
    def stylePaletteOnLeftUp(self, event: wx.MouseEvent):

        mousePos = event.GetPosition()

        if self.stylePalette.HasCapture():
            self.stylePalette.ReleaseMouse()

        if self.__config.isCanvasPosInRange(mousePos):
            self.__selectedCanvasPos = mousePos
            self.__selectedStyleIndex = self.__getStyleIndexByCanvasPos(mousePos)

            # wx.CallLater(1, self.__notifyLocation)

            self.__shouldRefresh = True
    
    def __notifySample(self):
        self.notify('sample', self.__sampleTemperature)
    
    def __notifyLocation(self, enabled=True):
        if enabled:
            location = self.__config.canvasPosToStyleLoc(self.__selectedCanvasPos)
        else:
            location = None
        self.notify('location', location)
    
    def stylePaletteOnRightDown(self, event: wx.MouseEvent):

        self.__selectedCanvasPos = None
        self.__selectedStyleIndex = None

        wx.CallLater(1, self.__notifyLocation, False)

        self.__shouldRefresh = True

    def stylePaletteOnMotion(self, event: wx.MouseEvent):

        mousePos = event.GetPosition()

        if mousePos != self.__currentMousePos:
            self.__currentMousePos = mousePos
            self.__shouldRefresh = True

            if self.__config.isCanvasPosInRange(mousePos):
                self.stylePalette.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
                self.__hoveredStyleIndex = self.__getStyleIndexByCanvasPos(mousePos)

                if event.Dragging() and event.LeftIsDown():
                    if self.__selectedCanvasPos:
                        self.__selectedCanvasPos = mousePos
                        self.__selectedStyleIndex = self.__getStyleIndexByCanvasPos(mousePos)

                        wx.CallLater(1, self.__notifyLocationDebounced)
                        self.__shouldRefresh = True
                
            else:
                self.stylePalette.SetCursor(wx.Cursor(wx.CURSOR_DEFAULT))
                self.stylePaletteOnLeftUp(event)
            
    def stylePaletteOnLeaveWindow(self, event: wx.MouseEvent):

        self.__currentMousePos = None

        self.stylePaletteOnLeftUp(event)

    def stylePaletteOnMouseWheel(self, event: wx.MouseEvent):
        
        shift = event.GetWheelRotation() / event.GetWheelDelta()
        self.__sampleTemperature = np.clip(self.__sampleTemperature + shift * 0.1, 0.1, 1.2)
        self.__shouldRefresh = True

    def stylePaletteOnPaint(self, event: wx.PaintEvent):

        try:
            dc = wx.BufferedPaintDC(self.stylePalette)
        except RuntimeError:
            return

        cfg = self.__config

        currentCanvasPos = self.__currentMousePos
        currentStyleLoc = None

        if currentCanvasPos and cfg.isCanvasPosInRange(currentCanvasPos):
            currentStyleLoc = cfg.canvasPosToStyleLoc(currentCanvasPos)

        # Background
        dc.SetBackground(wx.Brush(cfg.backgroundColor))
        dc.Clear()

        for tick in [-3, -2, -1, 0, 1, 2, 3]:
            # Axis/Grid Lines
            if tick == 0:
                dc.SetPen(wx.Pen(cfg.axisColor, cfg.axisLineWidth))
            else:
                dc.SetPen(wx.Pen(cfg.gridColor, cfg.gridLineWidth))
            lineX = int(cfg.scaleWidth * tick / 6 + cfg.width / 2)
            lineY = int(cfg.scaleHeight * -tick / 6 + cfg.height / 2)
            dc.DrawLine(lineX, cfg.gridTop - cfg.padding, lineX, cfg.gridBottom + cfg.padding)
            dc.DrawLine(cfg.gridLeft - cfg.padding, lineY, cfg.gridRight + cfg.padding, lineY)
        
            if tick != 0:
                # Tick Labels
                dc.SetTextForeground(cfg.gridColor)
                dc.DrawText(str(tick), cfg.gridLeft - 16, cfg.verticalCenter + -tick * cfg.verticalTick - 16)
                dc.DrawText(str(tick), cfg.horizontalCenter + tick * cfg.horizontalTick + 5, cfg.gridBottom)

        # Axis Labels
        dc.SetTextForeground(cfg.axisColor)
        dc.DrawText(cfg.axisLabelTop, (cfg.horizontalCenter + 4, cfg.gridTop - 16))
        dc.DrawText(cfg.axisLabelBottom, (cfg.horizontalCenter + 4, cfg.gridBottom))
        dc.DrawText(cfg.axisLabelLeft, (cfg.gridLeft - 30, cfg.verticalCenter))
        dc.DrawText(cfg.axisLabelRight, (cfg.gridRight + 4, cfg.verticalCenter))
        
        currentGroupStyle = None

        for styleIndex, (group, name, location, combination) in enumerate(self.__styleCombs):
            locX, locY = cfg.styleLocToCanvasPos(location)

            if styleIndex == self.__hoveredStyleIndex:
                linePen = wx.Pen(cfg.styleHoverLineColor, cfg.styleLineWidth)
                outlinePen = wx.Pen(cfg.styleHoverLineColor, cfg.styleHoverLineWidth)
                bgBrush = wx.Brush(cfg.styleHoverBgColor)
                labelFont = cfg.styleHoverFont
            else:
                linePen = wx.Pen(cfg.styleLineColor, cfg.styleLineWidth)
                outlinePen = linePen
                bgBrush = wx.Brush(cfg.styleBgColor)
                labelFont = cfg.styleFont

            if styleIndex == self.__selectedStyleIndex:
                outlinePen = wx.Pen(cfg.markPen.Colour, cfg.styleHoverLineWidth)

            # Style Name Background
            # dc.SetPen(wx.TRANSPARENT_PEN)
            # dc.SetBrush(bgBrush)
            # dc.DrawRectangle(locX - cfg.styleWidth // 2, locY - cfg.styleHeight // 2, cfg.styleWidth, cfg.styleHeight)

            for tick, color in enumerate(map(tuple, combination)):
                cX = int(locX - cfg.styleWidth / 2 + cfg.styleWidth * tick / 3)
                cY = int(locY - cfg.styleHeight / 2)
                cW = int(cfg.styleWidth / 3)
                cH = int(cfg.styleHeight)

                # Single Color
                dc.SetPen(wx.Pen(wx.Colour(color)))
                dc.SetBrush(wx.Brush(wx.Colour(color)))
                dc.DrawRectangle(cX, cY, cW, cH)

            # Split Line
            # lineX1 = locX - cfg.styleWidth // 2
            # lineX2 = locX + cfg.styleWidth // 2
            # dc.SetPen(wx.Pen(cfg.styleLineColor))
            # dc.DrawLine(lineX1, locY, lineX2, locY)

            # Outline
            topLeft = locX - cfg.styleWidth // 2, locY - cfg.styleHeight // 2
            dc.SetPen(outlinePen)
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.DrawRectangle(*topLeft, cfg.styleWidth, cfg.styleHeight)

            # Style Name
            # dc.SetFont(labelFont)
            # dc.SetTextForeground(wx.Colour(80, 80, 80))
            # dc.DrawText(f'{name}', locX - cfg.styleWidth // 2 + 2, locY - 2)
        
        if self.__selectedCanvasPos:
            posX, posY = self.__selectedCanvasPos
            locX, locY = cfg.canvasPosToStyleLoc(self.__selectedCanvasPos)
            shift = cfg.markSize // 2
            # text = f'({locX:.2f}, {locY:.2f}) T={self.__sampleTemperature:.1f}'
            dc.SetPen(cfg.markPen)
            dc.DrawLine(posX - shift, posY - shift, posX + shift, posY + shift)
            dc.DrawLine(posX + shift, posY - shift, posX - shift, posY + shift)
            # dc.SetFont(cfg.markFont)
            # dc.SetTextForeground(cfg.markBgColor)
            # dc.DrawText(text, posX + shift + 1, posY + shift + 1)
            # dc.SetTextForeground(cfg.markFgColor)
            # dc.DrawText(text, posX + shift, posY + shift)

        if currentStyleLoc:
            locX, locY = currentStyleLoc
            mouseX, mouseY = currentCanvasPos
            shiftX, shiftY = 8, 6

            # Style Location Label
            text = f'({locX:.2f}, {locY:.2f}) T={self.__sampleTemperature:.1f}'
            dc.SetFont(cfg.styleLocFont)
            dc.SetTextForeground(cfg.styleLocBgColor)
            dc.DrawText(text, mouseX + shiftX + 1, mouseY + shiftY + 1)
            dc.SetTextForeground(cfg.styleLocFgColor)
            dc.DrawText(text, mouseX + shiftX, mouseY + shiftY)
    
    def stylePaletteOnIdle(self, event: wx.IdleEvent):
        if self.__shouldRefresh:
            self.stylePalette.Refresh(False)
            self.__shouldRefresh = False