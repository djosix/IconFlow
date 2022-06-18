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

class EventManager:
    def __init__(self):
        self.__number = 0
        self.__channels = {}

    def __getChannelMapping(self, channel):
        assert '#' not in channel
        if channel not in self.__channels:
            self.__channels[channel] = {}
        return self.__channels[channel]
    
    def subscribe(self, channel, callback):
        mapping = self.__getChannelMapping(channel)
        handleId = f'{channel}#{self.__number}'
        self.__number += 1
        mapping[handleId] = callback
        return handleId
    
    def unsubscribe(self, handleId):
        channel = handleId.split('#')[0]
        mapping = self.__getChannelMapping(channel)
        if handleId in mapping:
            del mapping[handleId]
            return True
        return False
    
    def notify(self, channel, *args, **kwargs):
        mapping = self.__getChannelMapping(channel)
        if len(mapping) == 0:
            raise Exception('No subscriber in channel {}'.format(channel))
        for callback in mapping.values():
            callback(*args, **kwargs)

    def clearChannel(self, channel=None):
        self.__channels[channel] = {}

    def clearAll(self):
        self.__channels = {}
