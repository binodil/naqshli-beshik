
import uuid
import sys
from typing import Dict

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QCheckBox
from PyQt5.QtCore import pyqtSignal, QObject, Qt, pyqtSlot, QTimer

# https://stackoverflow.com/questions/56019273/how-can-i-get-more-input-text-in-pyqt5-inputdialog




class Task:
    def __init__(self, text:str):
        self.id = uuid.uuid4()
        self.text = text
    
    def __eq__(self, other):
        return self.id == other.id

class Item(QCheckBox):
    '''https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QCheckBox.html#qcheckbox'''

    signal_delete = pyqtSignal(Task)

    def __init__(self, task:Task, parent=None):
        super().__init__(task.text, parent=parent)
        self.task = task
        self.stateChanged.connect(self.on_state_changed)
        # delete timer value
        self.countdown_ms = 1000
    
    @pyqtSlot(int)
    def on_state_changed(self, val:int):
        print(val)
        QTimer.singleShot(self.countdown_ms, self.__on_delete)
        # set timer to be deleted in x seconds
    
    def __on_delete(self):
        self.signal_delete.emit(self.task)
        self.hide()
        self.deleteLater()


class Model(QObject):
    updated = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.tasks: Dict = {}
    
    def add(self, text:str):
        task = Task(text)
        item = Item(task)
        item.signal_delete.connect(self.delete)
        self.tasks[task.id] = task
        self.updated.emit(item)

    def delete(self, new_task:Task):
        # only task can collect to delete itself
        if new_task.id in self.tasks:
            del self.tasks[new_task.id]


class View(QWidget):
    signal_add_item = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.add_button = QPushButton("+", self)
        self.input_label = QLineEdit(self)
        main_layout = QVBoxLayout()
        add_layout = QHBoxLayout()
        add_layout.addWidget(self.input_label)
        add_layout.addWidget(self.add_button)
        main_layout.addLayout(add_layout)
        self.results = QWidget()
        res_layout = QVBoxLayout()
        res_layout.setAlignment(Qt.AlignTop)  # Tasks stick to top
        res_layout.setSpacing(3)  # Consistent spacing between tasks
        self.results.res_layout = res_layout
        self.results.setLayout(res_layout)
        main_layout.addWidget(self.results)

        self.setLayout(main_layout)

        self.add_button.clicked.connect(self.__on_add_item)

    def connect_signal_slots(self, model):
        # connect with model
        self.signal_add_item.connect(model.add)
        model.updated.connect(self.on_add_widget)

    def __on_add_item(self):
        # own things stay here
        text = self.input_label.text()
        self.input_label.setText("")
        self.signal_add_item.emit(text)
    
    def on_add_widget(self, item:Item):
        item.signal_delete.connect(self.adjustSize)
        self.results.res_layout.addWidget(item)


        

    


app = QApplication(sys.argv)
model = Model()
view = View()
view.connect_signal_slots(model)
view.show()
sys.exit(app.exec_())
