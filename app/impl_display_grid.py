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
from PIL import Image
from typing import List

from .event import EventManager
from .utils import bitmapToImage, imageToBitmap

class DisplayGridImplMixin(EventManager):
    previewSlots: List[wx.StaticBitmap]

    def __init__(self):

        self.subscribe('display', self.__updateSlots)
        self.subscribe('cache_and_preview', self.__previewHighResResult)

        self.__slotIndexDict = {}
        self.__shouldRefresh = {}

        for i, slot in enumerate(self.previewSlots.values()):
            slot: wx.StaticBitmap

            self.__slotIndexDict[id(slot)] = i
            self.__shouldRefresh[id(slot)] = False

            defaultImage = Image.new('RGB', tuple(slot.Size), (255, 255, 255))
            slot.SetBitmap(imageToBitmap(defaultImage))
            slot.SetCursor(wx.Cursor(wx.CURSOR_HAND))

        self.__highResResults = {}
        self.__sequenceNumber = 0
        self.__currentSlotId = None

    def __updateSlots(self, images: List[Image.Image]):

        self.__highResResults.clear()
        self.__sequenceNumber += 1

        for i, image in enumerate(images):
            self.previewSlots[i].SetBitmap(imageToBitmap(image))
            self.previewSlots[i].Refresh(False)

    def __previewHighResResult(self, highResResult, slotId, sequenceNumber):
        
        if sequenceNumber != self.__sequenceNumber:
            return
        
        self.__highResResults[slotId] = highResResult

        if slotId == self.__currentSlotId:
            self.__preview(slotId)

    def previewSlotsOnLeftDown(self, event: wx.MouseEvent):
    
        self.saveButtonOnButtonClick(event)

    def previewSlotsOnEnterWindow(self, event: wx.MouseEvent):
    
        self.__hoveringPreviewSlot(id(event.GetEventObject()))
    
    def previewSlotsOnMotion(self, event: wx.MouseEvent):

        self.__hoveringPreviewSlot(id(event.GetEventObject()))

    def __hoveringPreviewSlot(self, slotId):
        
        if self.__currentSlotId != slotId:
            self.__currentSlotId = slotId
            self.__preview(slotId)

    def __preview(self, slotId):

        if slotId in self.__highResResults:
            result = self.__highResResults[slotId]
        
        else:
            slot = self.previewSlots[self.__slotIndexDict[slotId]]
            result = bitmapToImage(slot.Bitmap)

            self.notify('upsample', result, slotId, self.__sequenceNumber)

        self.notify('preview', result)

    def previewSlotsOnLeaveWindow(self, event: wx.MouseEvent):
    
        self.__currentSlotId = None
        self.notify('preview', None)
    