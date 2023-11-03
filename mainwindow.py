# This Python file uses the following encoding: utf-8
import sys
import ast
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from PySide6.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from PySide6.QtCore import QTimer

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_MainWindow


class MainWindow(QMainWindow):
    currentGraph = set()
    supportVar = []

    smejmatrix = []
    dostmatrix = []
    vzaimmatrix = []
    primmatrix = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.listFunctions = [self.ui.check_smej, self.ui.check_dost, self.ui.check_vzaim,
                            self.ui.check_printgraph, self.ui.check_eiler, self.ui.check_minostPrim,
                            self.ui.check_components, self.ui.check_dekstra, self.ui.check_minostKruskal]
        self.listChecked = [0,0,0,0,0,0,0,0,0]
        self.currentMatrix = 1
        #Connections
        self.ui.button_addver.clicked.connect(self.addEdge)
        self.ui.button_delver.clicked.connect(self.delEdge)
        self.ui.button_addgraph.clicked.connect(self.addGraph)
        self.ui.button_cleargraph.clicked.connect(self.clear_currentGraph)
        self.ui.combo_for.currentIndexChanged.connect(self.updateEdit_for)
        self.ui.combo_to.currentIndexChanged.connect(self.updateEdit_to)
        self.ui.button_func.clicked.connect(self.executeFunctions)
        self.ui.button_changeTable.clicked.connect(self.updateMatrix)
        self.ui.check_minostPrim.clicked.connect(self.update_combobox_state)
        for i in range(0, len(self.listFunctions)):
            self.listFunctions[i].clicked.connect(self.updateFunc)
        # --------

    def generate_sorted_adjacency_matrix(self, graph, is_directed=False):
        sorted_nodes = sorted(set(node for edge in graph for node in edge[:2]))

        if is_directed:
            adjacency_matrix = np.zeros((len(sorted_nodes), len(sorted_nodes)))
            for edge in graph:
                i = sorted_nodes.index(edge[0])
                j = sorted_nodes.index(edge[1])
                weight = edge[2]
                adjacency_matrix[i][j] = weight
        else:
            adjacency_matrix = np.zeros((len(sorted_nodes), len(sorted_nodes)))
            for edge in graph:
                i = sorted_nodes.index(edge[0])
                j = sorted_nodes.index(edge[1])
                weight = edge[2]
                adjacency_matrix[i][j] = weight
                adjacency_matrix[j][i] = weight

        return adjacency_matrix
    def showError(self, typeError: str):
        self.ui.label_commandline.setText(f">> {typeError}")


    def showResult(self, typeResult: str):
        self.ui.label_commandline.setText(f">> {typeResult}")


    def addEdge(self):
        if(self.validationEdge() == 1):
            self.setSupportVar_edit()
            self.currentGraph.add(tuple(self.supportVar))
            self.update_currentGraph()
            self.showResult(f"Вершина {tuple(self.supportVar)} добавлена")
            self.clearSupportVar()
            return 1
        else:
            self.showError("Некорректная вершина")
            return 0


    def addGraph(self):
        localeVar = self.format_string(self.ui.edit_pastegraph.text())
        if self.validationGraph(localeVar):
            localeVar = ast.literal_eval(localeVar)
            self.currentGraph = self.currentGraph.union(localeVar)
            self.update_currentGraph()
            self.showResult("Граф объединён")
            return 1
        else:
            self.showError("Некорректный граф")
            return 0


    def delEdge(self):
        if(self.validationEdge() == 1):
            self.setSupportVar_edit()
            if tuple(self.supportVar) in self.currentGraph:
                self.currentGraph.remove(tuple(self.supportVar))
                self.showResult(f"Вершина {tuple(self.supportVar)} удалена")
                self.update_currentGraph()
            else:
                self.showResult(f"Вершина {tuple(self.supportVar)} не найдена")

            self.clearSupportVar()
            return 1
        else:
            self.showError("Некорректная вершина")
            return 0


    def setSupportVar_edit(self):
        self.supportVar.append(self.ui.edit_for.text())
        self.supportVar.append(self.ui.edit_to.text())
        self.supportVar.append(self.ui.spin_dist.value())

        return 1

    def clearSupportVar(self):
        self.supportVar.clear()
        self.supportVar = list()

        return 1

    def update_currentGraph(self):
        if len(self.currentGraph) == 0:
            self.ui.edit_thisgraph.setText("")
        else:
            self.ui.edit_thisgraph.setText(str(sorted(self.currentGraph)))

        return 1

    def clear_currentGraph(self):
        if len(self.currentGraph) != 0:
            self.currentGraph.clear()
            self.clearSupportVar()
            self.update_currentGraph()
            self.showResult("Текущий граф удален")
        else:
            self.showResult("Текущий граф пуст")

        return 1


    def format_string(self, input_str):
        pattern = re.compile(r"\(([^,]+),\s*([^,]+),\s*(\d+)\)")
        formatted_str = re.sub(pattern, r"('\1', '\2', \3)", input_str)
        formatted_str = re.sub(r"''", "'", formatted_str)
        return formatted_str


    def validationGraph(self, testGraph: str):
        try:
            tuple_set = eval(testGraph)
            if isinstance(tuple_set, set):
                if all(isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], str) and isinstance(item[1], str) and isinstance(item[2], int) for item in tuple_set):
                    return True
        except Exception as e:
            pass

        return False


    def validationEdge(self):
        if len(self.ui.edit_for.text()) == 0 or len(self.ui.edit_to.text()) == 0:
            return 0
        else:
            return 1


    def save_lastGraph(self):
        pass
    def updateEdit_for(self):
        self.ui.edit_for.setText(self.ui.combo_for.currentText())


    def updateEdit_to(self):
        self.ui.edit_to.setText(self.ui.combo_to.currentText())
    def updateFunc(self):
        for i in range(0, len(self.listFunctions)):
            self.listChecked[i] = self.listFunctions[i].isChecked()

    def executeFunctions(self):
        if self.listChecked[0]: #smej
            if len(self.currentGraph) == 0:
                self.showError("Граф пуст")
                return 0
            if self.ui.check_orient.isChecked() == True:
                self.smejmatrix = self.generate_sorted_adjacency_matrix(self.currentGraph, self.ui.check_orient.isChecked())
            else:
                self.smejmatrix = self.generate_sorted_adjacency_matrix(self.currentGraph, self.ui.check_orient.isChecked())
            self.showResult("Выполнено")
            self.updateMatrix()
        if self.listChecked[1]: #dost
            if len(self.currentGraph) == 0:
                self.showError("Граф пуст")
                return 0
            self.dostmatrix = self.generate_reachability_matrix(self.smejmatrix)
            self.showResult("Выполнено")
            self.updateMatrix()
        if self.listChecked[2]: #vzaim
            if len(self.currentGraph) == 0:
                self.showError("Граф пуст")
                return 0
            self.vzaimmatrix = self.generate_mutual_reachability_matrix(self.smejmatrix)
            self.showResult("Выполнено")
            self.updateMatrix()
        if self.listChecked[3]: #print graph
            if len(self.currentGraph) == 0:
                self.showError("Граф пуст")
                return 0
            self.printGraph(self.currentGraph, self.ui.check_orient.isChecked())
            self.showResult("Выполнено")
        if self.listChecked[4]: #eiler
            is_eulerian, is_semi_eulerian = self.is_eulerian(self.currentGraph, self.ui.check_orient.isChecked())
            if len(self.currentGraph) == 0:
                self.showError("Граф пуст")
                return 0
            self.ui.edit_eiler.setText(f"Эйлеровость, полуэйлеровость: {is_eulerian}, {is_semi_eulerian}")
            self.showResult("Выполнено")
        if self.listChecked[5]: #prim
            if self.ui.check_orient.isChecked():
                self.showError("Граф должен быть неориентированный (Прим)")
                return 0
            self.primmatrix = self.Prim(self.currentGraph, self.ui.combo_vertices.currentText())
            self.updateMatrix()
            self.showResult("Выполнено")
        if self.listChecked[6]:
            if len(self.currentGraph) == 0:
                self.showError("Граф пуст")
                return 0

            result = self.strong_components_weighted(self.currentGraph)
            if self.listChecked[4]:
                support_text = self.ui.edit_eiler.toPlainText()
                self.ui.edit_eiler.setText(support_text + f"; Компоненты сил. связности: {result}")
            else:
                self.ui.edit_eiler.setText(f"Компоненты сил. связности: {result}")
            self.showResult("Выполнено")

        if self.listChecked[7]:
            pass
        if self.listChecked[8]:
            pass
        if all(element == False for element in self.listChecked):
            self.showError("Ни одна функция не выбрана")
    def update_combobox_state(self):
        if len(self.currentGraph) == 0:
            self.showError("Граф пуст")
            return 0
        self.ui.combo_vertices.clear()
        vertices = set()
        for edge in self.currentGraph:
            vertices.add(edge[0])
            vertices.add(edge[1])
        self.ui.combo_vertices.addItems(sorted(vertices))
        self.ui.combo_vertices.setEnabled(self.ui.check_minostPrim.isChecked())
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

    def generate_reachability_matrix(self, adjacency_matrix):
        num_vertices = len(adjacency_matrix)
        reach_matrix = np.array(adjacency_matrix)

        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    reach_matrix[i][j] = reach_matrix[i][j] or (reach_matrix[i][k] and reach_matrix[k][j])

        np.fill_diagonal(reach_matrix, 1)
        return (reach_matrix > 0).astype(int)

    def generate_mutual_reachability_matrix(self, adjacency_matrix):
        reach_matrix = self.generate_reachability_matrix(adjacency_matrix)
        mutual_reach_matrix = np.logical_and(reach_matrix, np.transpose(reach_matrix))

        np.fill_diagonal(mutual_reach_matrix, 1)
        return mutual_reach_matrix.astype(int)


    def is_eulerian(self, graph, is_directed=False):
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)

        for edge in graph:
            out_degree[edge[0]] += 1
            in_degree[edge[1]] += 1

        odd_degree_count = sum(1 for vertex in set(in_degree.keys()) | set(out_degree.keys()) if (in_degree[vertex] + out_degree[vertex]) % 2 != 0)

        is_eulerian = odd_degree_count == 0
        is_semi_eulerian = odd_degree_count == 2 if not is_directed else False

        return is_eulerian, is_semi_eulerian


    def strong_components_weighted(self, graph_edges):
        G = nx.DiGraph()
        G.add_weighted_edges_from(graph_edges)

        return sorted(list(nx.strongly_connected_components(G)))


    def updateMatrix(self):
        selected_matrices = []

        if self.ui.check_smej.isChecked() and len(self.smejmatrix) != 0:
            self.smejmatrix[self.smejmatrix > 0] = 1
            self.smejmatrix[self.smejmatrix <= 0] = 0

            selected_matrices.append(("Матрица смежности", self.smejmatrix.astype(int)))

        if self.ui.check_dost.isChecked() and len(self.dostmatrix) != 0:
            selected_matrices.append(("Матрица достижимости", self.dostmatrix))

        if self.ui.check_vzaim.isChecked() and len(self.vzaimmatrix) != 0:
            selected_matrices.append(("Матрица взаимной достижимости", self.vzaimmatrix))

        if self.ui.check_minostPrim.isChecked() and len(self.primmatrix) != 0:
            selected_matrices.append(("Минимальное остовное дерево (Прим)", self.primmatrix))
        if selected_matrices:
            current_matrix, current_data = selected_matrices[self.currentMatrix - 1]
            self.fill_table(self.ui.table_main, current_data)
            self.ui.edit_currentTable.setText(current_matrix)

            self.currentMatrix = (self.currentMatrix % len(selected_matrices)) + 1

    def printGraph(self, graph, is_directed: bool):
        if is_directed:
            G = nx.DiGraph()
            G.add_weighted_edges_from(graph)

            pos = nx.circular_layout(G)
            labels = nx.get_edge_attributes(G, 'weight')

            nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=12, font_color="red")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

            plt.title("Ориентированный граф")
            plt.show()
        else:
            G = nx.Graph()
            G.add_weighted_edges_from(graph)

            min_spanning_tree = nx.minimum_spanning_tree(G)

            colors = nx.coloring.greedy_color(G, strategy='largest_first')

            pos = nx.circular_layout(G)
            labels = nx.get_edge_attributes(G, 'weight')

            node_colors = [colors[node] for node in G.nodes()]
            edge_colors = ['red' if edge in min_spanning_tree.edges() else 'gray' for edge in G.edges()]
            nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=8, font_color="black", edge_color=edge_colors)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

            plt.title("Неориентированный граф с минимальным остовным деревом")
            plt.show()


    def Prim(self, graph_edges, start_node):
        selected_nodes = {start_node}

        selected_edges = []

        while len(selected_nodes) < len(set(node for edge in graph_edges for node in edge[:2])):
            possible_edges = [edge for edge in graph_edges if edge[0] in selected_nodes and edge[1] not in selected_nodes or
                              edge[1] in selected_nodes and edge[0] not in selected_nodes]

            min_edge = min(possible_edges, key=lambda x: x[2])
            selected_edges.append(min_edge)
            selected_nodes.add(min_edge[0] if min_edge[1] in selected_nodes else min_edge[1])


        mst_list = []
        addonlist = [start_node]

        for i in range(len(selected_edges)):
            element1 = str()
            for element in selected_edges[i][:2]:
                if element not in addonlist:
                    element1 = element
            suplist = [i + 1, sorted(addonlist.copy()), selected_edges[i][:-1], element1, selected_edges[i][2]]
            mst_list.append(suplist)
            addonlist.append(element1)

        return mst_list

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
