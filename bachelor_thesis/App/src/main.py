# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uiOfApp.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import time

from PyQt5 import QtCore, QtGui, QtWidgets

from stl.mesh import Mesh
from PIL.ImageQt import ImageQt

from torchvision import transforms
from PIL import ImageDraw
import vtkplotlib as vpl



import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL  import  Image
import numpy as np

from bachelor_thesis.Trainning.teeth.yoloV3 import YOLOv3
import torch.optim as optim
import torch
from bachelor_thesis.Trainning.teeth.yoloV3 import box_convertor

from bachelor_thesis.Trainning.teeth.yoloV3 import cells_to_bboxes, non_max_suppression

IMAGE_SIZE = 512
NUM_CLASSES = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

img_path = "D:\\Renders\\Render_annotation_ver20\\000001_0\\input\\x=0y=0z=0.png"
yolo_v3_path = "/bachelor_thesis/models_checkpoint\\yolo\\checkpoint-245.pth.tar"


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
)



model = YOLOv3(num_classes=NUM_CLASSES).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

load_checkpoint(
    "C:\\Users\\mosva\\Documents\\Code\\bachelor_trainning\\pytorch_google_colab\\bachelor_thesis\\models_checkpoint\\yolo\\checkpoint-" + str(
        248) + ".pth.tar", model, optimizer, LEARNING_RATE
)
model.eval()

with torch.no_grad():
    scaled_anchors = (
            torch.tensor(ANCHORS)
            * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(DEVICE)


def show_predict_yolo(img, img_draw, thresh, iou_thresh ):

    print("preditcting")
    img_tensor = img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_tensor)
        bboxes = [[] for _ in range(img_tensor.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box


    #display image

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )

    image = img_tensor[0].permute(1,2,0).detach().cpu()
    numpy_image = image.numpy()
    boxes = box_convertor(numpy_image, nms_boxes)
    print("Let's")

    for box in boxes:
        print(box)
        img_draw.rectangle(box,fill=None,width=1,outline="yellow")






class FigureAndButton(QtWidgets.QWidget):
    def __init__(self,parent = None):

        QtWidgets.QWidget.__init__(self, parent)

        # Go for a vertical stack layout.
        vbox = QtWidgets.QVBoxLayout(parent)
        self.setLayout(vbox)

        # Create the figure
        self.figure = vpl.QtFigure()

        # Create a button and attach a callback.
        self.button = QtWidgets.QPushButton("Select a teeth")
        self.button.released.connect(self.button_pressed_cb)

        # QtFigures are QWidgets and are added to layouts with `addWidget`
        vbox.addWidget(self.figure)
        vbox.addWidget(self.button)


    def button_pressed_cb(self):
        """Plot commands can be called in callbacks. The current working
        figure is still self.figure and will remain so until a new
        figure is created explicitly. So the ``fig=self.figure``
        arguments below aren't necessary but are recommended for
        larger, more complex scenarios.
        """

        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 
         'c:\\',"Teeth files (*.stl)")[0].replace('/','\\')
        print(fname)


        mesh = Mesh.from_file(fname)

        # Plot the mesh
        vpl.mesh_plot(mesh,color='silver')
        fig = vpl.gcf()
        fig.background_color = "black"
        

        # Randomly place a ball.


        # Reposition the camera to better fit to the balls.
        vpl.reset_camera(self.figure)

        # Without this the figure will not redraw unless you click on it.
        self.figure.update()


    def show(self):
        # The order of these two are interchangeable.
        super().show()
        self.figure.show()


    def closeEvent(self, event):
        """This isn't essential. VTK, OpenGL, Qt and Python's garbage
        collect all get in the way of each other so that VTK can't
        clean up properly which causes an annoying VTK error window to
        pop up. Explicitly calling QtFigure's `closeEvent()` ensures
        everything gets deleted in the right order.
        """
        self.figure.closeEvent(event)




class Ui_Dialog(object):


    def analyze(self):
        mask_rcnn_path = '/bachelor_thesis/models_checkpoint\\mask-rcnn\\mask-rcnn-model-all61.pt'
        # --------------- INPUT --------------------

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        teeth_screenshot = Image.fromarray(np.array(vpl.screenshot_fig()).copy())
        loader = transforms.Compose([transforms.ToTensor()])

        # --------------- INPUT -------------------- 

        # --------------- Resizing for Neural Network -------------------- 

        teeth_screenshot_mask_rcnn_input = teeth_screenshot.resize((256,256))
        teeth_screenshot_yolo_input = teeth_screenshot.resize((512,512))
        teeth_screenshot_yolo_input_draw = ImageDraw.Draw(teeth_screenshot_yolo_input)

        # --------------- Resizing for Neural Network --------------------

        # -----------------MASK-RCNN-----------------------------------------------

        
        modelMaskRcnn = torch.load(mask_rcnn_path)
        modelMaskRcnn.eval()


        teeth_screenshot_mask_rcnn_input_tensor = loader(teeth_screenshot_mask_rcnn_input)
        prediction_mask_rcnn = modelMaskRcnn([teeth_screenshot_mask_rcnn_input_tensor.to(device)])
        teeth_screenshot_mask_rcnn_input_numpy = np.array(teeth_screenshot_mask_rcnn_input)

        for i in range(len(prediction_mask_rcnn[0]['masks'])):
        # iterate over masks
            mask = prediction_mask_rcnn[0]['masks'][i, 0]
            mask = mask.mul(255).byte().cpu().numpy()
            teeth_screenshot_mask_rcnn_input_numpy[ mask > 1] +=  np.array([60,0,0],dtype='uint8')

        mask_rcnn_out = Image.fromarray(teeth_screenshot_mask_rcnn_input_numpy)
        image_draw_rect = ImageDraw.Draw(mask_rcnn_out)

        for box in prediction_mask_rcnn[0]['boxes'].byte().cpu().numpy():
            x1,y1,x2,y2 = box
            image_draw_rect.rectangle([x1,y1,x2,y2],fill=None,width=1,outline="red")

        mask_rcnn_rezised_out = Image.fromarray(np.array(mask_rcnn_out).copy()).resize((480,480))
        
        self.widget_2.setPixmap(
            QtGui.QPixmap.fromImage(
                ImageQt(mask_rcnn_rezised_out)
            )
        )
       
        time.sleep(4)
        del  image_draw_rect, mask_rcnn_out, teeth_screenshot_mask_rcnn_input_numpy, prediction_mask_rcnn, teeth_screenshot_mask_rcnn_input_tensor , modelMaskRcnn

        # -----------------MASK-RCNN-----------------------------------------------

        # ----------------YoloV3 -------------------

        img = np.array(teeth_screenshot_yolo_input)
        augmentations = test_transforms(image=img)
        img = augmentations["image"]
        show_predict_yolo( img,teeth_screenshot_yolo_input_draw, 0.6, 0.5)
        teeth_screenshot_yolo_input = teeth_screenshot_yolo_input.resize((250,250))
        self.widget_2.setPixmap(
            QtGui.QPixmap.fromImage(
                ImageQt(teeth_screenshot_yolo_input)
            )
        )




    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1448, 728)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(1190, 50, 181, 161))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 160, 111))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout.addWidget(self.checkBox)
        self.checkBox_2 = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox_2.setObjectName("checkBox_2")
        self.verticalLayout.addWidget(self.checkBox_2)
        self.checkBox_5 = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox_5.setObjectName("checkBox_5")
        self.verticalLayout.addWidget(self.checkBox_5)
        self.checkBox_4 = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox_4.setObjectName("checkBox_4")
        self.verticalLayout.addWidget(self.checkBox_4)
        self.checkBox_3 = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox_3.setObjectName("checkBox_3")
        self.verticalLayout.addWidget(self.checkBox_3)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(1240, 280, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.widget = FigureAndButton(Dialog)
        self.widget.setGeometry(QtCore.QRect(40, 60, 512, 512))
        self.widget_2 = QtWidgets.QLabel(Dialog)
        self.widget_2.setGeometry(QtCore.QRect(620, 46, 512, 512))
        self.widget_2.setObjectName("widget_2")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(50, 30, 47, 13))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(630, 30, 131, 16))
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(1230, 240, 91, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton.released.connect(self.analyze)

    

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)



    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "Option"))
        self.checkBox.setText(_translate("Dialog", "Mask-Rcnn Masks"))
        self.checkBox_2.setText(_translate("Dialog", "Mask-RCNN  Boxes ( red )"))
        self.checkBox_5.setText(_translate("Dialog", " Yolo Boxes ( green )"))
        self.checkBox_4.setText(_translate("Dialog", "Unet Landmarks per teeth"))
        self.checkBox_3.setText(_translate("Dialog", "Unet Landmarks"))
        self.pushButton.setText(_translate("Dialog", "Analyze"))
        self.label.setText(_translate("Dialog", "Teeth"))
        self.label_2.setText(_translate("Dialog", "Teeth analyzed"))
        self.pushButton_2.setText(_translate("Dialog", "Select  a teeth"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())


