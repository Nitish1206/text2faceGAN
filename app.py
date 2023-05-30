from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from image_generator import ImageGenerator
from main_window import Ui_MainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt
import cv2
from torchvision.utils import save_image



# The female has pretty high cheekbones and an oval face. Her hair is black.

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__()
        self.name_str=0
        self.gen_model=ImageGenerator()
        self.setupUi(self)
        self.generator.clicked.connect(self.show_image)

    def show_image(self):
        self.name_str+=1
        if self.sentence_input.text():
            text_=self.sentence_input.text()

            frame=self.gen_model.gen(text_)
            save_image(frame, "temp/img"+str(self.name_str)+".png")


            frame=cv2.imread( "temp/img"+str(self.name_str)+".png")
            frame=cv2.resize(frame,(640,480))

            faceimbytesPerLine = 3 * frame.shape[1]
            image = QImage(frame, frame.shape[1], frame.shape[0],faceimbytesPerLine ,QImage.Format.Format_BGR888)
            qpix_img  = QPixmap.fromImage(image)
            # set a scaled pixmap to a w x h window keeping its aspect ratio 
            self.image_frame.setPixmap(qpix_img.scaled(self.image_frame.size(),
                                                Qt.AspectRatioMode.KeepAspectRatio))
            # self.image_frame.setPixmap(qpix_img)
        else:
            print("empty str")
            
        self.sentence_input.clear()
        

def main():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print("Closing Application ...")
        # exit()

if __name__ == '__main__':
    main()