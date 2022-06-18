# -*- coding: utf-8 -*-

from .resource import Paths

###########################################################################
## Python code generated with wxFormBuilder (version 3.10.1-0-g8feb16b3)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class TemplateFrame
###########################################################################

class TemplateFrame ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'IconFlow', pos = wx.DefaultPosition, size = wx.Size( 1089,764 ), style = wx.CAPTION|wx.CLOSE_BOX|wx.SYSTEM_MENU|wx.TAB_TRAVERSAL )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
        self.SetBackgroundColour( wx.Colour( 240, 240, 240 ) )

        mainSizer = wx.BoxSizer( wx.VERTICAL )

        topPanelSizer = wx.BoxSizer( wx.HORIZONTAL )

        self.stylePalette = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
        self.stylePalette.SetMinSize( wx.Size( 542,542 ) )

        topPanelSizer.Add( self.stylePalette, 0, wx.EXPAND|wx.TOP|wx.RIGHT|wx.LEFT, 5 )

        sketchPanelSizer = wx.BoxSizer( wx.VERTICAL )

        sketchToolbarSizer = wx.WrapSizer( wx.HORIZONTAL, wx.WRAPSIZER_DEFAULT_FLAGS )

        self.newButton = wx.BitmapButton( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, wx.BU_AUTODRAW|0 )

        self.newButton.SetBitmap( wx.Bitmap( Paths.NEW_BUTTON, wx.BITMAP_TYPE_ANY ) )
        self.newButton.SetToolTip( 'New Sketch' )

        sketchToolbarSizer.Add( self.newButton, 1, 0, 5 )

        self.openButton = wx.BitmapButton( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, wx.BU_AUTODRAW|0 )

        self.openButton.SetBitmap( wx.Bitmap( Paths.OPEN_BUTTON, wx.BITMAP_TYPE_ANY ) )
        self.openButton.SetToolTip( 'Load Sketch' )

        sketchToolbarSizer.Add( self.openButton, 0, 0, 5 )

        self.saveButton = wx.BitmapButton( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, wx.BU_AUTODRAW|0 )

        self.saveButton.SetBitmap( wx.Bitmap( Paths.SAVE_BUTTON, wx.BITMAP_TYPE_ANY ) )
        self.saveButton.SetToolTip( 'Save Image' )

        sketchToolbarSizer.Add( self.saveButton, 0, 0, 5 )

        self.pencilButton = wx.BitmapButton( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, wx.BU_AUTODRAW|0 )

        self.pencilButton.SetBitmap( wx.Bitmap( Paths.PENCIL_BUTTON, wx.BITMAP_TYPE_ANY ) )
        self.pencilButton.SetToolTip( 'Pencil' )

        sketchToolbarSizer.Add( self.pencilButton, 0, 0, 5 )

        self.eraserButton = wx.BitmapButton( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, wx.BU_AUTODRAW|0 )

        self.eraserButton.SetBitmap( wx.Bitmap( Paths.ERASER_BUTTON, wx.BITMAP_TYPE_ANY ) )
        self.eraserButton.SetToolTip( 'Eraser' )

        sketchToolbarSizer.Add( self.eraserButton, 0, 0, 5 )

        self.undoButton = wx.BitmapButton( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, wx.BU_AUTODRAW|0 )

        self.undoButton.SetBitmap( wx.Bitmap( Paths.UNDO_BUTTON, wx.BITMAP_TYPE_ANY ) )
        self.undoButton.SetToolTip( 'Undo' )

        sketchToolbarSizer.Add( self.undoButton, 0, wx.ALIGN_BOTTOM, 5 )

        self.redoButton = wx.BitmapButton( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, wx.BU_AUTODRAW|0 )

        self.redoButton.SetBitmap( wx.Bitmap( Paths.REDO_BUTTON, wx.BITMAP_TYPE_ANY ) )
        self.redoButton.SetToolTip( 'Redo' )

        sketchToolbarSizer.Add( self.redoButton, 0, 0, 5 )


        sketchPanelSizer.Add( sketchToolbarSizer, 0, wx.TOP, 5 )

        self.sketchCanvas = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 512,512 ), 0 )
        self.sketchCanvas.SetMinSize( wx.Size( 512,512 ) )
        self.sketchCanvas.SetMaxSize( wx.Size( 512,512 ) )

        sketchPanelSizer.Add( self.sketchCanvas, 0, wx.FIXED_MINSIZE|wx.BOTTOM|wx.RIGHT, 0 )


        topPanelSizer.Add( sketchPanelSizer, 0, 0, 5 )


        mainSizer.Add( topPanelSizer, 0, 0, 5 )

        displayGridSizer = wx.GridSizer( 1, 8, 0, 0 )

        self.previewSlots = {}
        self.previewSlots[0] = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 128,128 ), 0 )
        self.previewSlots[0].SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
        self.previewSlots[0].SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

        displayGridSizer.Add( self.previewSlots[0], 0, 0, 5 )

        self.previewSlots[1] = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 128,128 ), 0 )
        self.previewSlots[1].SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

        displayGridSizer.Add( self.previewSlots[1], 0, 0, 5 )

        self.previewSlots[2] = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 128,128 ), 0 )
        self.previewSlots[2].SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

        displayGridSizer.Add( self.previewSlots[2], 0, 0, 5 )

        self.previewSlots[3] = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 128,128 ), 0 )
        self.previewSlots[3].SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

        displayGridSizer.Add( self.previewSlots[3], 0, 0, 5 )

        self.previewSlots[4] = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 128,128 ), 0 )
        self.previewSlots[4].SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

        displayGridSizer.Add( self.previewSlots[4], 0, 0, 5 )

        self.previewSlots[5] = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 128,128 ), 0 )
        self.previewSlots[5].SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

        displayGridSizer.Add( self.previewSlots[5], 0, 0, 5 )

        self.previewSlots[6] = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 128,128 ), 0 )
        self.previewSlots[6].SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

        displayGridSizer.Add( self.previewSlots[6], 0, 0, 5 )

        self.previewSlots[7] = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 128,128 ), 0 )
        self.previewSlots[7].SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

        displayGridSizer.Add( self.previewSlots[7], 0, 0, 5 )


        mainSizer.Add( displayGridSizer, 0, wx.EXPAND|wx.TOP|wx.BOTTOM|wx.LEFT, 5 )


        self.SetSizer( mainSizer )
        self.Layout()
        self.statusBar = self.CreateStatusBar( 1, wx.STB_SIZEGRIP, wx.ID_ANY )

        self.Centre( wx.BOTH )

        # Connect Events
        self.stylePalette.Bind( wx.EVT_LEAVE_WINDOW, self.stylePaletteOnLeaveWindow )
        self.stylePalette.Bind( wx.EVT_LEFT_DOWN, self.stylePaletteOnLeftDown )
        self.stylePalette.Bind( wx.EVT_LEFT_UP, self.stylePaletteOnLeftUp )
        self.stylePalette.Bind( wx.EVT_MOTION, self.stylePaletteOnMotion )
        self.stylePalette.Bind( wx.EVT_MOUSEWHEEL, self.stylePaletteOnMouseWheel )
        self.stylePalette.Bind( wx.EVT_PAINT, self.stylePaletteOnPaint )
        self.stylePalette.Bind( wx.EVT_RIGHT_DOWN, self.stylePaletteOnRightDown )
        self.newButton.Bind( wx.EVT_BUTTON, self.newButtonOnButtonClick )
        self.openButton.Bind( wx.EVT_BUTTON, self.openButtonOnButtonClick )
        self.saveButton.Bind( wx.EVT_BUTTON, self.saveButtonOnButtonClick )
        self.pencilButton.Bind( wx.EVT_BUTTON, self.pencilButtonOnButtonClick )
        self.eraserButton.Bind( wx.EVT_BUTTON, self.eraserButtonOnButtonClick )
        self.undoButton.Bind( wx.EVT_BUTTON, self.undoButtonOnButtonClick )
        self.redoButton.Bind( wx.EVT_BUTTON, self.redoButtonOnButtonClick )
        self.sketchCanvas.Bind( wx.EVT_ENTER_WINDOW, self.sketchCanvasOnEnterWindow )
        self.sketchCanvas.Bind( wx.EVT_LEAVE_WINDOW, self.sketchCanvasOnLeaveWindow )
        self.sketchCanvas.Bind( wx.EVT_LEFT_DOWN, self.sketchCanvasOnLeftDown )
        self.sketchCanvas.Bind( wx.EVT_LEFT_UP, self.sketchCanvasOnLeftUp )
        self.sketchCanvas.Bind( wx.EVT_MOTION, self.sketchCanvasOnMotion )
        self.sketchCanvas.Bind( wx.EVT_PAINT, self.sketchCanvasOnPaint )
        self.previewSlots[0].Bind( wx.EVT_ENTER_WINDOW, self.previewSlotsOnEnterWindow )
        self.previewSlots[0].Bind( wx.EVT_LEAVE_WINDOW, self.previewSlotsOnLeaveWindow )
        self.previewSlots[0].Bind( wx.EVT_LEFT_DOWN, self.previewSlotsOnLeftDown )
        self.previewSlots[0].Bind( wx.EVT_MOTION, self.previewSlotsOnMotion )
        self.previewSlots[1].Bind( wx.EVT_ENTER_WINDOW, self.previewSlotsOnEnterWindow )
        self.previewSlots[1].Bind( wx.EVT_LEAVE_WINDOW, self.previewSlotsOnLeaveWindow )
        self.previewSlots[1].Bind( wx.EVT_LEFT_DOWN, self.previewSlotsOnLeftDown )
        self.previewSlots[1].Bind( wx.EVT_MOTION, self.previewSlotsOnMotion )
        self.previewSlots[2].Bind( wx.EVT_ENTER_WINDOW, self.previewSlotsOnEnterWindow )
        self.previewSlots[2].Bind( wx.EVT_LEAVE_WINDOW, self.previewSlotsOnLeaveWindow )
        self.previewSlots[2].Bind( wx.EVT_LEFT_DOWN, self.previewSlotsOnLeftDown )
        self.previewSlots[2].Bind( wx.EVT_MOTION, self.previewSlotsOnMotion )
        self.previewSlots[3].Bind( wx.EVT_ENTER_WINDOW, self.previewSlotsOnEnterWindow )
        self.previewSlots[3].Bind( wx.EVT_LEAVE_WINDOW, self.previewSlotsOnLeaveWindow )
        self.previewSlots[3].Bind( wx.EVT_LEFT_DOWN, self.previewSlotsOnLeftDown )
        self.previewSlots[3].Bind( wx.EVT_MOTION, self.previewSlotsOnMotion )
        self.previewSlots[4].Bind( wx.EVT_ENTER_WINDOW, self.previewSlotsOnEnterWindow )
        self.previewSlots[4].Bind( wx.EVT_LEAVE_WINDOW, self.previewSlotsOnLeaveWindow )
        self.previewSlots[4].Bind( wx.EVT_LEFT_DOWN, self.previewSlotsOnLeftDown )
        self.previewSlots[4].Bind( wx.EVT_MOTION, self.previewSlotsOnMotion )
        self.previewSlots[5].Bind( wx.EVT_ENTER_WINDOW, self.previewSlotsOnEnterWindow )
        self.previewSlots[5].Bind( wx.EVT_LEAVE_WINDOW, self.previewSlotsOnLeaveWindow )
        self.previewSlots[5].Bind( wx.EVT_LEFT_DOWN, self.previewSlotsOnLeftDown )
        self.previewSlots[5].Bind( wx.EVT_MOTION, self.previewSlotsOnMotion )
        self.previewSlots[6].Bind( wx.EVT_ENTER_WINDOW, self.previewSlotsOnEnterWindow )
        self.previewSlots[6].Bind( wx.EVT_LEAVE_WINDOW, self.previewSlotsOnLeaveWindow )
        self.previewSlots[6].Bind( wx.EVT_LEFT_DOWN, self.previewSlotsOnLeftDown )
        self.previewSlots[6].Bind( wx.EVT_MOTION, self.previewSlotsOnMotion )
        self.previewSlots[7].Bind( wx.EVT_ENTER_WINDOW, self.previewSlotsOnEnterWindow )
        self.previewSlots[7].Bind( wx.EVT_LEAVE_WINDOW, self.previewSlotsOnLeaveWindow )
        self.previewSlots[7].Bind( wx.EVT_LEFT_DOWN, self.previewSlotsOnLeftDown )
        self.previewSlots[7].Bind( wx.EVT_MOTION, self.previewSlotsOnMotion )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def stylePaletteOnLeaveWindow( self, event ):
        event.Skip()

    def stylePaletteOnLeftDown( self, event ):
        event.Skip()

    def stylePaletteOnLeftUp( self, event ):
        event.Skip()

    def stylePaletteOnMotion( self, event ):
        event.Skip()

    def stylePaletteOnMouseWheel( self, event ):
        event.Skip()

    def stylePaletteOnPaint( self, event ):
        event.Skip()

    def stylePaletteOnRightDown( self, event ):
        event.Skip()

    def newButtonOnButtonClick( self, event ):
        event.Skip()

    def openButtonOnButtonClick( self, event ):
        event.Skip()

    def saveButtonOnButtonClick( self, event ):
        event.Skip()

    def pencilButtonOnButtonClick( self, event ):
        event.Skip()

    def eraserButtonOnButtonClick( self, event ):
        event.Skip()

    def undoButtonOnButtonClick( self, event ):
        event.Skip()

    def redoButtonOnButtonClick( self, event ):
        event.Skip()

    def sketchCanvasOnEnterWindow( self, event ):
        event.Skip()

    def sketchCanvasOnLeaveWindow( self, event ):
        event.Skip()

    def sketchCanvasOnLeftDown( self, event ):
        event.Skip()

    def sketchCanvasOnLeftUp( self, event ):
        event.Skip()

    def sketchCanvasOnMotion( self, event ):
        event.Skip()

    def sketchCanvasOnPaint( self, event ):
        event.Skip()

    def previewSlotsOnEnterWindow( self, event ):
        event.Skip()

    def previewSlotsOnLeaveWindow( self, event ):
        event.Skip()

    def previewSlotsOnLeftDown( self, event ):
        event.Skip()

    def previewSlotsOnMotion( self, event ):
        event.Skip()






























