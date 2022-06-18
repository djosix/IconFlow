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
import time
import queue
import threading
from PIL import Image
from typing import Dict

from .event import EventManager
from .model import IconFlow
from .utils import spawn, queue_clear
from .config import IMAGE_SIZE


class Event(wx.PyEvent):
    EVT_ID = wx.NewId()

    def __init__(self, *data):
        super().__init__()
        self.SetEventType(self.EVT_ID)
        self.__data = data
    
    def GetData(self):
        return self.__data


class ModelInferenceImplMixin(EventManager):
    previewSlots: Dict[int, wx.StaticBitmap]
    statusBar: wx.StatusBar

    def __init__(self):

        self.__model = IconFlow()

        self.subscribe('sketch', self.__provideSketch)
        self.subscribe('sample', self.__provideTemperature)
        self.subscribe('location', self.__provideStyleLocation)
        self.subscribe('upsample', self.__provideLowResResult)

        self.__stopped = False

        self.__sketchQueue = queue.Queue(1)
        self.__locationQueue = queue.Queue(1)
        self.__decodeEvent = threading.Event()
        self.__upsampleQueue = queue.Queue(8)

        self.__numberOfSamples = len(self.previewSlots)
        self.__sketchImage = None
        self.__sketchImageHighRes = None
        self.__sketchContent = None
        self.__styleNoises = None
        self.__styleVectors = None

        self.Connect(-1, -1, Event.EVT_ID, self.__eventHandler)

        spawn(self.__styleWorker)
        spawn(self.__contentWorker)
        spawn(self.__decodeWorker)
        spawn(self.__upsampleWorker)

        self.__contentTime = None
        self.__styleTime = None
        self.__decodeTime = None
        self.__upsampleTime = None
    
    def __updateStatusBar(self):

        times = {}

        if self.__contentTime is not None:
            times['content'] = self.__contentTime

        if self.__styleTime is not None:
            times['style'] = self.__styleTime

        if self.__decodeTime is not None:
            times['output'] = self.__decodeTime
        
        if self.__upsampleTime is not None:
            times['upsample'] = self.__upsampleTime

        if times:
            msgText = ', '.join([
                f'[{name}] {elapsed:.2f}s'
                for name, elapsed in times.items()
            ])
            self.statusBar.SetLabelText(msgText)

    def __del__(self):

        self.__stopped = True

    def __eventHandler(self, event: Event):
        channel, *args = event.GetData()
        self.notify(channel, *args)

    def __styleWorker(self):

        while not self.__stopped:
            location = self.__locationQueue.get()
            noises = self.__styleNoises

            if noises is None or location is None:
                self.__styleVectors = None
            else:
                startTime = time.time()
                self.__styleVectors = self.__model.get_styles(noises, location)
                self.__styleTime = time.time() - startTime
                self.__updateStatusBar()
            
            self.__decodeEvent.set()

    def __contentWorker(self):
        
        while not self.__stopped:
            sketch = self.__sketchQueue.get()
            self.__sketchImageHighRes = sketch
            sketch = sketch.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
            self.__sketchImage = sketch

            if sketch is None:
                self.__sketchContent = None
            else:
                try:
                    startTime = time.time()
                    self.__sketchContent = self.__model.get_embeddings(sketch)
                    self.__contentTime = time.time() - startTime
                    self.__updateStatusBar()
                except:
                    import traceback
                    print(traceback.format_exc())
                    # continue
            
            self.__decodeEvent.set()

    def __decodeWorker(self):
        
        while not self.__stopped:
            self.__decodeEvent.wait()
            self.__decodeEvent.clear()
            
            if self.__sketchImage is None:
                images = [
                    Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
                ] * self.__numberOfSamples

            elif self.__styleVectors is None:
                images = [
                    self.__sketchImage.convert('RGB')
                ] * self.__numberOfSamples

            else:
                sketchContent = self.__sketchContent

                if sketchContent is None:
                    return
                
                sketchContent = sketchContent.expand(self.__styleVectors.shape[0], -1, -1, -1)

                startTime = time.time()
                images = self.__model.decode(sketchContent, self.__styleVectors)
                self.__decodeTime = time.time() - startTime
                self.__updateStatusBar()

            wx.PostEvent(self, Event('display', images))

    def __upsampleWorker(self):

        while not self.__stopped:

            lowResResult, slotId, sequenceNumber = self.__upsampleQueue.get()
            highResSketch = self.__sketchImageHighRes

            if highResSketch is None:
                continue

            if self.__styleVectors is None:
                self.__upsampleTime = None
                highResResult = highResSketch
                
            else:
                startTime = time.time()
                highResResult = self.__model.upsample(lowResResult, highResSketch)
                self.__upsampleTime = time.time() - startTime

            wx.PostEvent(self, Event('cache_and_preview', highResResult, slotId, sequenceNumber))

    def __provideSketch(self, sketch):
        
        if sketch is not None:
            sketch = sketch.convert('L')

        queue_clear(self.__sketchQueue)
        self.__sketchQueue.put(sketch)

    def __provideTemperature(self, temperature):
        
        if temperature is None:
            self.__styleNoises = None

        else:
            self.__styleNoises = self.__model.sample_noises(
                self.__numberOfSamples, temperature)

    def __provideStyleLocation(self, location):
        
        if location is not None:
            locationX, locationY = location
            location = (locationX / 3, locationY / 3)
        
        queue_clear(self.__locationQueue)
        self.__locationQueue.put(location)

    def __provideLowResResult(self, lowResResult, slotId, sequenceNumber):

        self.__upsampleQueue.put((lowResResult, slotId, sequenceNumber))
        