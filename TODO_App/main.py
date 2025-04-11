
import uuid
import time
import sys
import csv
from typing import List
from enum import Enum, auto

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QLayout
from PyQt5.QtCore import pyqtSignal, QObject, Qt, pyqtSlot, QTimer

# https://stackoverflow.com/questions/56019273/how-can-i-get-more-input-text-in-pyqt5-inputdialog

# I will use Model to store activities and all business items
# I will use QWidgets or GUI to emit signals for it

DEFAULT_DB_PATH = "database.csv"


class StatusEnum(Enum):
    TODO = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()


class Task:
    def __init__(self, text:str, status:StatusEnum):
        self.id = uuid.uuid4()
        self.text = text
        self.status = status
    
    def __eq__(self, other):
        return self.id == other.id


class Model(QObject):
    updated = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.tasks: Dict = {}
    
    def add(self, task:Task):
        self.tasks[task.id] = task
        self.updated.emit(list(self.tasks.values()))

    def delete(self, new_task:Task):
        # only task can collect to delete itself
        if new_task.id in self.tasks:
            del self.tasks[new_task.id]
            self.updated.emit(list(self.tasks.values()))

    def update(self, new_task:Task):
        self.tasks[new_task.id].text = new_task.text
        self.tasks[new_task.id].status = new_task.status
        self.updated.emit(list(self.tasks.values()))



class Controller(QObject):
    signal_update_view = pyqtSignal(object)

    def __init__(self, model):
        super().__init__()
        self.view = None
        self.model = model
    
    def set_view(self, view):
        self.view = view
        self.model.updated.connect(self.on_model_updated)
        # self.signal_update_view.connect(self.view.on_model_updated)
    
    def on_add_item(self, text):
        item = Task(text, status="regular")
        self.model.add(item)
    
    def on_edit_item(self, task:Task):
        self.model.update(task)

    def on_delete_item(self, task:Task):
        self.model.delete(task)
    
    def on_model_updated(self, tasks):
        # shall we create a View 
        # messy here
        final_tasks = []
        for task in tasks:
            task = Item(task)
            task.signal_delete.connect(self.on_delete_item)
            # task.signal_edit_item.connect(self.on_edit_item)
            final_tasks.append(task)

        self.signal_update_view.emit(final_tasks)  # emit to View


class View(QWidget):
    signal_add_item = pyqtSignal(object)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.add_button = QPushButton("Add", self)
        self.input_label = QLineEdit(self)
        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.add_button)
        self.results = QWidget()
        res_layout = QVBoxLayout()
        res_layout.setAlignment(Qt.AlignTop)  # Tasks stick to top
        res_layout.setSpacing(5)  # Consistent spacing between tasks
        self.results.res_layout = res_layout
        self.results.setLayout(res_layout)
        layout.addWidget(self.results)

        # Prevent results from expanding unnecessarily
        layout.setStretch(0, 0)  # Input_label doesn’t stretch
        layout.setStretch(1, 0)  # Add_button doesn’t stretch
        layout.setStretch(2, 1)  # Results takes extra space, but tasks won’t grow
        self.setLayout(layout)

        self.add_button.clicked.connect(self.on_add_item)
        self.signal_add_item.connect(self.controller.on_add_item)

    def on_add_item(self):
        # own things stay here
        text = self.input_label.text()
        item = Item(Task(text, StatusEnum.TODO))
        item.signal_delete.connect(self.adjustSize)
        self.results.res_layout.addWidget(item)

        


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
        

    


app = QApplication(sys.argv)
model = Model()
controller = Controller(model)
view = View(controller)
controller.set_view(view)
view.show()



sys.exit(app.exec_())