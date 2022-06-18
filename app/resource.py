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

import os

MODULE_DIR = os.path.dirname(__file__)
RESOURCE_DIR = os.path.join(MODULE_DIR, 'resource_files')

def resource_path(*path):
    return os.path.join(RESOURCE_DIR, *path)

class Paths:
    COLOR_IMAGE_SCALE = resource_path('color_image_scale.pkl')
    
    DRAWING_CURSOR = resource_path('cursors', 'draw.png')
    ERASING_CURSOR = resource_path('cursors', 'erase.png')

    NEW_BUTTON = resource_path('buttons', 'new.png')
    OPEN_BUTTON = resource_path('buttons', 'open.png')
    SAVE_BUTTON = resource_path('buttons', 'save.png')
    PENCIL_BUTTON = resource_path('buttons', 'pencil.png')
    ERASER_BUTTON = resource_path('buttons', 'eraser.png')
    UNDO_BUTTON = resource_path('buttons', 'undo.png')
    REDO_BUTTON = resource_path('buttons', 'redo.png')
