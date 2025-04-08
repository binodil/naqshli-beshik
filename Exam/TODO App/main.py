
import uuid
import time
import sys
import csv
from typing import List
from enum import Enum, auto

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import pyqtSignal, QObject, Qt

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
        self.signal_update_view.connect(self.view.on_model_updated)
    
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
            task = TaskUI(task)
            task.signal_delete.connect(self.on_delete_item)
            task.signal_edit_item.connect(self.on_edit_item)
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
        self.signal_add_item.emit(text)

    def _remove_widgets(self):
        # ineffective. smelly. Maybe refresh or redraw? or delete the layout
        while self.results.res_layout.count():
            item = self.results.res_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()  # explicitly deleting it

    def on_model_updated(self, database:List):
        self._remove_widgets()
        for task in database:
            self.results.res_layout.addWidget(task)


class TaskUI(QWidget):
    # make it robust, ui friendly

    signal_delete = pyqtSignal(object)
    signal_edit_item = pyqtSignal(object)

    def __init__(self, task):
        self.task = task
        super().__init__()
        self.line_edit = QLineEdit(task.text, self)
        self.line_edit.setReadOnly(True)
        # self.status_label = QLabel(task.status, self)
        edit_button = QPushButton("Edit", self)
        delete_button = QPushButton("Delete", self)
        core_layout = QVBoxLayout()
        core_layout.addWidget(self.line_edit)
        # core_layout.addWidget(self.status_label)
        main_layout = QHBoxLayout()
        main_layout.addWidget(edit_button)  # left side
        main_layout.addLayout(core_layout)
        main_layout.addWidget(delete_button)
        self.setLayout(main_layout)

        #####connections#####
        edit_button.clicked.connect(self.on_edit_button_clicked)
        delete_button.clicked.connect(self.on_delete_button)

    def on_edit_button_clicked(self):
        if self.line_edit.isReadOnly():  # allow to edit
            self.line_edit.setReadOnly(False)
        
        else:
            # when user press again the button after the modification
            self.task.text = self.line_edit.text()
            self.signal_edit_item.emit(self.task)

    def on_delete_button(self):
        # let the model know to remove this task instance
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