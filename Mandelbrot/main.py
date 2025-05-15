
from PyQt5.QtCore import QThread, QSize, pyqtSignal, QMutex, QWaitCondition, QMutexLocker, Qt, QPoint
from PyQt5.QtWidgets import QWidget, QApplication

from PyQt5.QtGui import QImage, qRgb, QPainter, qFuzzyCompare, QColor, QPixmap
from PIL import Image
import numpy as np


class RenderThread(QThread):
    '''renders the Mandelbrot set.'''
    signal_renderedImage = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.mutex = QMutex()
        self.condition = QWaitCondition()

        self.restart = False
        self.abort = False
        self.ColormapSize = 256

    def render(self, centerX:float, centerY:float, scaleFactor:float, resultSize:QSize, devicePixelRatio:float):
        # start the rendering, stop the rendering, re-start rendering
        with QMutexLocker(self.mutex):
            self.centerX = centerX
            self.centerY = centerY
            self.scaleFactor = scaleFactor
            self.devicePixelRatio = devicePixelRatio
            # assert self.devicePixelRatio == 0, "devicePixelRatio is 0"
            self.resultSize = resultSize
            print("render is called")
            print(devicePixelRatio)

            if (not self.isRunning()):
                self.start()
            else:
                self.restart = True
                self.condition.wakeOne()  # maybe wake all?
    #---------------
    def run(self):
        while True:
            self.mutex.lock()
            # Storing the member variables in local variables allows us to minimize the amout of code that needs to be protected by a mutex.
            # integer copies when init new variable
            # list is pointed to the original list
            devicePixelRatio = self.devicePixelRatio
            resultSize = self.resultSize * devicePixelRatio if devicePixelRatio != 0 else self.resultSize
            requestedScaleFactor = self.scaleFactor
            print(requestedScaleFactor, devicePixelRatio)
            scaleFactor = requestedScaleFactor / devicePixelRatio if devicePixelRatio != 0 else requestedScaleFactor
            centerX = self.centerX
            centerY = self.centerY
            self.mutex.unlock()

            #################################
            halfWidth = resultSize.width() // 2
            halfHeight = resultSize.height() // 2
            

            pil_image = Image.new("RGB", (resultSize.width(), resultSize.height()))
            # image = QImage(resultSize, QImage.Format_RGB32)   # QImage image(resultSize, QImage::Format_RGB32);
            # image.setDevicePixelRatio(devicePixelRatio)
            
            NumPasses = 8
            pass_ = 0
            # The rendering is done in NumPasses (8) iterations, with increasing precision.
            while (pass_ < NumPasses):
                print("pass value: ", pass_)
                MaxIterations = (1 << (2 * pass_ + 6)) + 32  # bitwise operation 
                print("MaxIterations is ", MaxIterations)
                Limit = 4
                allBlack = True
                for y in range(-halfHeight, halfHeight):
                    if self.restart:
                        break
                    if self.abort:
                        return
                    ay = centerY + (y * scaleFactor)
                    for x in range(-halfWidth, halfWidth):
                        ax = centerX + (x * scaleFactor)
                        a1, b1 = ax, ay
                        num_iterations = 0

                        while num_iterations < MaxIterations:
                            num_iterations += 1
                            a2 = (a1 * a1) - (b1 * b1) + ax
                            b2 = (2 * a1 * b1) + ay
                            if (a2 * a2) + (b2 * b2) > Limit:
                                break
                            num_iterations += 1
                            a1 = (a2 * a2) - (b2 * b2) + ax
                            b1 = (2 * a2 * b2) + ay
                            if (a1 * a1) + (b1 * b1) > Limit:
                                break

                        if num_iterations < MaxIterations:
                            # color_tuple = (
                            #     num_iterations % self.ColormapSize, 
                            #     (num_iterations**2) % self.ColormapSize, 
                            #     (num_iterations**3) % self.ColormapSize)
                            
                            color_tuple = (
                                255 - num_iterations % self.ColormapSize, 
                                255 - (num_iterations * 2) % self.ColormapSize, 
                                255 - (num_iterations * 4) % self.ColormapSize)
                            # color_tuple = (255, 0, 0)
                            # if itevations are very small, it means they are not stable. dark red means not stable, white means stable
                            # color_tuple = (255 - int(255 * num_iterations/MaxIterations), 0, 0)
                            # color_tuple = (
                            #     num_iterations % self.ColormapSize, 
                            #     0, #num_iterations % self.ColormapSize, 
                            #     0, #num_iterations % self.ColormapSize,
                            #     # num_iterations % 255,
                            #     )
                            pil_image.putpixel((x + halfWidth, y + halfHeight), color_tuple)
                            # print("Color is ", num_iterations % self.ColormapSize)
                            # image.setPixel(x + halfWidth, y + halfHeight,
                            #             self.colormap[num_iterations % self.ColormapSize])
                            # image.setPixel(x + halfWidth, y + halfHeight, 3)
                            allBlack = False
                        else:
                            color_black = (0, 0, 0)
                            color_white = (255, 255, 255)
                            color = color_black  # black means stability, singularity like black holes
                            pil_image.putpixel((x + halfWidth, y + halfHeight), color)
            #-----------
                if allBlack and pass_ == 0:
                    pass_ = 4
                else:
                    if not self.restart:
                        self.signal_renderedImage.emit((pil_image, requestedScaleFactor))  # Simulated signal
                    pass_ += 1
                
            print("while loop of drawing is completed")
            self.mutex.lock()
            if not self.restart:
                print("Waiting...")
                self.condition.wait(self.mutex)  # Wait for a signal
                print("Waiting is terminated.")
            self.restart = False
            self.mutex.unlock()
            ###########################################

# CONSTANTS
DefaultCenterX = -0.637011
DefaultCenterY = -0.0395159
DefaultScale = 0.00403897

ZoomInFactor = 0.8
ZoomOutFactor = 1 / ZoomInFactor
ScrollStep = 20
#=============

class MandelbrotWidget(QWidget):
    '''
    shows the Mandelbrot set on screen and lets the user zoom and scroll.
    '''

    def __init__(self, centerX=DefaultCenterX, centerY=DefaultCenterY, pixmapScale=DefaultScale, curScale=DefaultScale, ZoomInFactor=ZoomInFactor, ZoomOutFactor=ZoomOutFactor, ScrollStep=ScrollStep):
        super().__init__()
        self.thread = RenderThread()
        self.pixmap = QPixmap()
        self.pixmapOffset = QPoint()
        self.lastDragPos = QPoint()
        ##############################
        self.centerX = centerX
        self.centerY = centerY
        self.pixmapScale = pixmapScale
        self.curScale = curScale
        self.ZoomInFactor = ZoomInFactor
        self.ZoomOutFactor = ZoomOutFactor
        self.ScrollStep = ScrollStep

        # CONNECTINOS
        self.thread.signal_renderedImage.connect(self.updatePixmap) # Connected

        self.setWindowTitle("Mandelbrot")
        self.setCursor(Qt.CrossCursor)
        
        self.resize(550, 400)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if (self.pixmap.isNull()):
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Rendering initial image, please wait...")
            return
        #-----------------------------
        print("(self.curScale, self.pixmapScale): ", (self.curScale, self.pixmapScale))
        # if (qFuzzyCompare(self.curScale, self.pixmapScale)):
        if round(self.curScale, 5) == round(self.pixmapScale, 5):
            print("Drawing with qFuzzy if compare")
            painter.drawPixmap(self.pixmapOffset, self.pixmap)
        else:

            print("Drawing with previewPixmap: round(self.pixmap.devicePixelRatioF(), 2): ", round(self.pixmap.devicePixelRatioF(), 2))
            previewPixmap = self.pixmap if round(self.pixmap.devicePixelRatioF(), 2) == 1 else self.pixmap.scaled(self.pixmap.size() / self.pixmap.devicePixelRatioF(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaleFactor = self.pixmapScale / self.curScale
            newWidth = int(previewPixmap.width() * scaleFactor)
            newHeight = int(previewPixmap.height() * scaleFactor)
            newX = self.pixmapOffset.x() + (previewPixmap.width() - newWidth) / 2
            newY = self.pixmapOffset.y() + (previewPixmap.height() - newHeight) / 2

            painter.save()
            painter.translate(newX, newY)
            painter.scale(scaleFactor, scaleFactor)
            inverted_transform, invertible = painter.transform().inverted()
            if not invertible:
                print("Transform is not invertible")
                return
            exposed = inverted_transform.mapRect(self.rect()).adjusted(-1, -1, 1, 1)

            painter.drawPixmap(exposed, previewPixmap, exposed)
            painter.restore()
        
        #-----------------------------
        text = "Use mouse wheel or the '+' and '-' keys to zoom. Press and hold left mouse button to scroll."
        metrics = painter.fontMetrics()
        textWidth = metrics.horizontalAdvance(text)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 127))
        painter.drawRect((self.width() - textWidth) // 2 - 5, 0, textWidth + 10, metrics.lineSpacing() + 5)
        painter.setPen(Qt.white)
        painter.drawText((self.width() - textWidth) // 2, metrics.leading() + metrics.ascent(), text)

    def resizeEvent(self, event):
        self.thread.render(self.centerX, self.centerY, self.curScale, self.size(), self.pixmap.devicePixelRatioF())

    def keyPressEvent(self, event):
        # switch (event->key()) 
        key = event.key()  # MAYBE BUG
        if key == Qt.Key_Plus:
            self.zoom(self.ZoomInFactor)
        elif key == Qt.Key_Minus:
            self.zoom(self.ZoomOutFactor)
        elif key == Qt.Key_Left:
            self.scroll(-self.ScrollStep, 0)
        elif key == Qt.Key_Right:
            self.scroll(+self.ScrollStep, 0)
        elif key == Qt.Key_Down:
            self.scroll(0, -self.ScrollStep)
        elif key == Qt.Key_Up:
            self.scroll(0, +self.ScrollStep)
        else:
            super().keyPressEvent(event)
    
    def wheelEvent(self, event):
        numDegrees = event.angleDelta().y() / 8
        numSteps = numDegrees / 15
        self.zoom(self.ZoomInFactor ** numSteps)

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton):
            self.lastDragPos = event.pos()

    def mouseMoveEvent(self, event):
        print('mouseMoveEvent')
        if (event.buttons() and Qt.LeftButton):
            self.pixmapOffset += event.pos() - self.lastDragPos
            self.lastDragPos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.LeftButton):
            self.pixmapOffset += event.pos() - self.lastDragPos
            self.lastDragPos = QPoint()

            pixmapSize = self.pixmap.size() / self.pixmap.devicePixelRatioF()
            deltaX = int((self.width() - pixmapSize.width()) / 2 - self.pixmapOffset.x())
            deltaY = int((self.height() - pixmapSize.height()) / 2 - self.pixmapOffset.y())
            self.scroll(deltaX, deltaY)
    # Slot

    def updatePixmap(self, result:object):
        (image, scaleFactor) = result

        if not self.lastDragPos.isNull():
            print("no drawing. cancellling")
            return
        img_array = np.array(image)
        height, width, channels = img_array.shape
        bytes_per_line = channels * width
        qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimage)
        self.pixmapOffset = QPoint()
        self.lastDragPos = QPoint()
        self.pixmapScale = scaleFactor
        self.update()
        print("drawing new pixmap!")

    def zoom(self, zoomFactor):
        self.curScale *= zoomFactor
        print("Zoom is applied so ", self.curScale)
        self.update()
        self.thread.render(self.centerX, self.centerY, self.curScale, self.size(), self.pixmap.devicePixelRatioF())

    def scroll(self, deltaX, deltaY):
        self.centerX += deltaX * self.curScale
        self.centerY += deltaY * self.curScale
        print("Scroll is applied so centerX and centerY are ", self.centerX, self.centerY)
        self.update()
        self.thread.render(self.centerX, self.centerY, self.curScale, self.size(), self.pixmap.devicePixelRatioF())






if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    widget = MandelbrotWidget()
    widget.show()
    sys.exit(app.exec_())
