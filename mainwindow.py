# This Python file uses the following encoding: utf-8
import sys
import ast
from graph_functions import *
from PySide6.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from PySide6.QtCore import QFile

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_MainWindow


class MainWindow(QMainWindow):
    # containers
    current_graph = set()
    support_list = []
    adjacency_matrix = []
    reachability_matrix = []
    mutual_reachability_matrix = []
    mst_step_list = []
    shortest_paths_list = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.listFunctions = [self.ui.check_smej, self.ui.check_dost, self.ui.check_vzaim,
                              self.ui.check_printgraph, self.ui.check_eiler, self.ui.check_minostPrim,
                              self.ui.check_components, self.ui.check_dekstra, self.ui.check_minostKruskal]
        self.listChecked = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.currentMatrix = 1
        # Connections
        self.ui.button_addver.clicked.connect(self.add_edge)
        self.ui.button_delver.clicked.connect(self.remove_edge)
        self.ui.button_addgraph.clicked.connect(self.add_graph)
        self.ui.button_cleargraph.clicked.connect(self.clear_current_graph)
        self.ui.combo_for.currentIndexChanged.connect(self.update_edit_for)
        self.ui.combo_to.currentIndexChanged.connect(self.update_edit_to)
        self.ui.button_func.clicked.connect(self.execute_functions)
        self.ui.button_changeTable.clicked.connect(self.update_matrices)
        self.ui.check_minostPrim.clicked.connect(self.update_mst_combobox)
        self.ui.check_dekstra.clicked.connect(self.update_paths_combobox)
        for i in range(0, len(self.listFunctions)):
            self.listFunctions[i].clicked.connect(self.update_func)
        # --------

    def show_message(self, message: str):
        self.ui.label_commandline.setText(f">> {message}")

    def add_edge(self):
        if self.validation_edge():
            self.set_support_list_edit()
            self.current_graph.add(tuple(self.support_list))
            self.update_current_graph()
            self.show_message(f"Вершина {tuple(self.support_list)} добавлена")
            self.clear_support_list()
            return 1
        else:
            self.show_message("Некорректная вершина")
            return 0

    def add_graph(self):
        locale_var = format_string(self.ui.edit_pastegraph.text())
        if validation_graph(locale_var):
            locale_var = ast.literal_eval(locale_var)
            self.current_graph = self.current_graph.union(locale_var)
            self.update_current_graph()
            self.show_message("Граф объединён")
            return 1
        else:
            self.show_message("Некорректный граф")
            return 0

    def remove_edge(self):
        if self.validation_edge():
            self.set_support_list_edit()
            if tuple(self.support_list) in self.current_graph:
                self.current_graph.remove(tuple(self.support_list))
                self.show_message(f"Вершина {tuple(self.support_list)} удалена")
                self.update_current_graph()
            else:
                self.show_message(f"Вершина {tuple(self.support_list)} не найдена")

            self.clear_support_list()
            return 1
        else:
            self.show_message("Некорректная вершина")
            return 0

    def set_support_list_edit(self):
        self.support_list.append(self.ui.edit_for.text())
        self.support_list.append(self.ui.edit_to.text())
        self.support_list.append(self.ui.spin_dist.value())

        return 1

    def clear_support_list(self):
        self.support_list.clear()
        self.support_list = list()

        return 1

    def update_current_graph(self):
        if len(self.current_graph) == 0:
            self.ui.edit_thisgraph.setText("")
        else:
            self.ui.edit_thisgraph.setText(str(sorted(self.current_graph)))

        return 1

    def clear_current_graph(self):
        if len(self.current_graph) != 0:
            self.current_graph.clear()
            self.clear_support_list()
            self.clear_matrices()
            self.update_matrices()
            self.update_current_graph()
            self.show_message("Текущий граф удален")
        else:
            self.show_message("Текущий граф пуст")

        return 1

    def clear_matrices(self):
        self.mst_step_list.clear()
        self.shortest_paths_list.clear()
        self.mutual_reachability_matrix.clear()

    def validation_edge(self):
        if len(self.ui.edit_for.text()) == 0 or len(self.ui.edit_to.text()) == 0:
            return 0
        else:
            return 1

    def save_last_graph(self):
        pass

    def update_edit_for(self):
        self.ui.edit_for.setText(self.ui.combo_for.currentText())

    def update_edit_to(self):
        self.ui.edit_to.setText(self.ui.combo_to.currentText())

    def update_func(self):
        for i in range(0, len(self.listFunctions)):
            self.listChecked[i] = self.listFunctions[i].isChecked()

    def execute_functions(self):
        if self.listChecked[0]:  # smej
            if len(self.current_graph) == 0:
                self.show_message("Граф пуст")
                return 0
            if self.ui.check_orient.isChecked():
                self.adjacency_matrix = generate_sorted_adjacency_matrix(self.current_graph,
                                                                         self.ui.check_orient.isChecked())
            else:
                self.adjacency_matrix = generate_sorted_adjacency_matrix(self.current_graph,
                                                                         self.ui.check_orient.isChecked())
            self.show_message("Выполнено")
            self.update_matrices()
        if self.listChecked[1]:  # dost
            if len(self.current_graph) == 0:
                self.show_message("Граф пуст")
                return 0
            self.reachability_matrix = generate_reachability_matrix(self.adjacency_matrix)
            self.show_message("Выполнено")
            self.update_matrices()
        if self.listChecked[2]:  # vzaim
            if len(self.current_graph) == 0:
                self.show_message("Граф пуст")
                return 0
            self.mutual_reachability_matrix = generate_mutual_reachability_matrix(self.adjacency_matrix)
            self.show_message("Выполнено")
            self.update_matrices()
        if self.listChecked[3]:  # print graph
            if len(self.current_graph) == 0:
                self.show_message("Граф пуст")
                return 0
            show_graph(self.current_graph, self.ui.check_orient.isChecked())
            self.show_message("Выполнено")
        if self.listChecked[4]:  # eiler
            is_eulerian, is_semi_eulerian = is_eulerians(self.current_graph, self.ui.check_orient.isChecked())
            if len(self.current_graph) == 0:
                self.show_message("Граф пуст")
                return 0
            self.ui.edit_eiler.setText(f"Эйлеровость, полуэйлеровость: {is_eulerian}, {is_semi_eulerian}")
            self.show_message("Выполнено")
        if self.listChecked[5]:  # prim
            if self.ui.check_orient.isChecked():
                self.show_message("Граф должен быть неориентированный (Прим)")
                return 0
            self.mst_step_list = create_mst_steps(self.current_graph, self.ui.combo_vertices.currentText())
            self.update_matrices()
            self.show_message("Выполнено")
        if self.listChecked[6]:
            if len(self.current_graph) == 0:
                self.show_message("Граф пуст")
                return 0

            result = strong_components_weighted(self.current_graph)
            sorted_sets = [sorted(s) for s in result]
            sorted_list = sorted(sorted_sets, key=lambda s: s[0])

            if self.listChecked[4]:
                support_text = self.ui.edit_eiler.toPlainText()
                self.ui.edit_eiler.setText(support_text + f"; Компоненты сил. связности: {sorted_list}")
            else:
                self.ui.edit_eiler.setText(f"Компоненты сил. связности: {sorted_list}")
            self.show_message("Выполнено")

        if self.listChecked[7]:
            if len(self.current_graph) == 0:
                self.show_message("Граф пуст")
                return 0
            self.shortest_paths_list = create_shortest_paths(self.current_graph,
                                                             self.ui.combo_vertices_2.currentText())
            self.update_matrices()
            self.show_message("Выполнено")
        if self.listChecked[8]:
            pass
        if all(element == False for element in self.listChecked):
            self.show_message("Ни одна функция не выбрана")

    def update_mst_combobox(self):
        if len(self.current_graph) == 0:
            self.show_message("Граф пуст")
            return 0
        self.ui.combo_vertices.clear()
        vertices = set()
        for edge in self.current_graph:
            vertices.add(edge[0])
            vertices.add(edge[1])
        self.ui.combo_vertices.addItems(sorted(vertices))
        self.ui.combo_vertices.setEnabled(self.ui.check_minostPrim.isChecked())

    def update_paths_combobox(self):
        if len(self.current_graph) == 0:
            self.show_message("Граф пуст")
            return 0
        self.ui.combo_vertices_2.clear()
        vertices = set()
        for edge in self.current_graph:
            vertices.add(edge[0])
            vertices.add(edge[1])
        self.ui.combo_vertices_2.addItems(sorted(vertices))
        self.ui.combo_vertices_2.setEnabled(self.ui.check_dekstra.isChecked())

    def fill_table(self, table, data):
        table.setRowCount(len(data))
        table.setColumnCount(len(data[0]))
        table.setVerticalHeaderLabels([''] * len(data))
        table.setHorizontalHeaderLabels([''] * len(data[0]))

        for i, row in enumerate(data):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

    def update_matrices(self):
        selected_matrices = []

        if self.ui.check_smej.isChecked() and len(self.adjacency_matrix) != 0:
            self.adjacency_matrix[self.adjacency_matrix > 0] = 1
            self.adjacency_matrix[self.adjacency_matrix <= 0] = 0

            selected_matrices.append(("Матрица смежности", self.adjacency_matrix.astype(int)))

        if self.ui.check_dost.isChecked() and len(self.reachability_matrix) != 0:
            selected_matrices.append(("Матрица достижимости", self.reachability_matrix))

        if self.ui.check_vzaim.isChecked() and len(self.mutual_reachability_matrix) != 0:
            selected_matrices.append(("Матрица взаимной достижимости", self.mutual_reachability_matrix))

        if self.ui.check_minostPrim.isChecked() and len(self.mst_step_list) != 0:
            selected_matrices.append(("Минимальное остовное дерево (Прим)", self.mst_step_list))

        if self.ui.check_dekstra.isChecked() and len(self.shortest_paths_list) != 0:
            selected_matrices.append(
                (f"Кратчайшие пути из вершины {self.ui.combo_vertices_2.currentText()}", self.shortest_paths_list))

        if selected_matrices:
            current_matrix, current_data = selected_matrices[self.currentMatrix - 1]
            self.fill_table(self.ui.table_main, current_data)
            self.ui.edit_currentTable.setText(current_matrix)

            self.currentMatrix = (self.currentMatrix % len(selected_matrices)) + 1


if __name__ == "__main__":
    app = QApplication(sys.argv)

    style_file = QFile("style.css")
    style_file.open(QFile.ReadOnly | QFile.Text)
    style = str(style_file.readAll(), encoding='utf-8')
    app.setStyleSheet(style)

    widget = MainWindow()
    widget.show()

    sys.exit(app.exec())
