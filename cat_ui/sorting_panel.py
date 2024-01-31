import functools

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QGridLayout
from PyQt5.QtWidgets import QListWidget, QLabel
from PyQt5.QtGui import QColor, QFont
from cat_ui.bertopic_model import TopicModel


class Document:
    def __init__(self, title, body, true_label, is_train=False):
        self.title = title
        self.body = body
        self.true_label = true_label
        self.is_train = is_train

    def getTitle(self):
        return self.title

    def getBody(self):
        return self.body

    def getString(self):
        body = self.body.replace('\n', '\\n')  # otherwise multi-line emails take a lot of space
        if self.true_label == -1:
            string = f'{self.title}|{body}'
        else:
            string = f'{self.true_label}|{self.title}|{body}'
        return string

    def getLabel(self):
        return self.true_label

    def isTrain(self):
        return self.is_train

    @classmethod
    def saveDocumentsTitlesAsCSV(cls, documents, classified_results, save_path):
        with open(save_path, 'w') as outfile:
            strs = []
            for i in range(len(documents)):
                document = documents[i]
                strs.append(f'{document.getTitle()}, {classified_results[i]}')
            outfile.write('\n'.join(strs))


class TextSelector(QWidget):
    def __init__(self, on_index_clicked, identity):
        super().__init__()

        self.on_index_clicked = on_index_clicked
        self.identity = identity

        # Setting up the QVBoxLayout
        self.layout = QVBoxLayout(self)

        # Creating the QListWidget and adding items
        self.list_widget = QListWidget(self)
        self.list_widget.itemClicked.connect(self.onClick)

        # Adding the QListWidget to the layout
        self.layout.addWidget(self.list_widget)

        self.setLayout(self.layout)

        self.docIdToIndex = {}
        self.indexToDocId = []

    def onClick(self, item):
        index = self.list_widget.currentRow()
        self.on_index_clicked(self, index, item)

    def getId(self):
        return self.identity

    def currentIndex(self):
        return self.list_widget.currentRow()

    def setCurrentIndex(self, i):
        if i == -1:
            self.list_widget.clearSelection()
        elif i >= 0:
            self.list_widget.setCurrentIndex(i)
        else:
            raise ValueError(f'attempt to set current index to {i}')

    def isAnythingSelected(self):
        return bool(self.getListWidget().selectedItems())

    def getListWidget(self):
        return self.list_widget

    def addDoc(self, docId, doc):
        i = self.list_widget.count()
        self.list_widget.addItem(doc.getString())
        self.docIdToIndex[docId] = i
        self.indexToDocId.append(docId)

    def setDocColor(self, docId, qColor):
        item = self.list_widget.item(self.docIdToIndex[docId])
        item.setBackground(qColor)

    def removeDoc(self, docId):
        # selectedDocId = self.getSelectedDocId()

        index = self.docIdToIndex[docId]
        self.list_widget.takeItem(index)
        self.indexToDocId.pop(index)

        # remove the docId from self.docIdToIndex and subtract 1 from every document following the index
        self.docIdToIndex.pop(docId)
        for i in range(index, len(self.indexToDocId)):
            curDocId = self.indexToDocId[i]
            self.docIdToIndex[curDocId] -= 1

        # if docId == selectedDocId:
        #     self.list_widget.clearSelection()

    def getSelectedDocId(self):
        if self.isAnythingSelected():
            selected = self.indexToDocId[self.list_widget.currentRow()]
        else:
            selected = None
        return selected


class SortingPanel(QWidget):
    def __init__(self, documents, num_categories):
        super().__init__()

        # Set up the grid layout
        self.grid = QGridLayout()
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.font().setPointSize(20)
        self.retrain_button = QPushButton("retrain and predict", self)
        self.save_button = QPushButton("save results", self)
        self.retrain_button.clicked.connect(self.retrainAndPredict)
        self.save_button.clicked.connect(self.saveResults)
        self.num_categories = num_categories

        self.selectorDict = {}
        for i in range(2):
            for j in range(self.num_categories):
                index = i * self.num_categories + j
                text_selector = TextSelector(self.onSelectorIndexClicked, index)
                self.selectorDict[index] = text_selector
                self.grid.addWidget(text_selector, i, j)
        self.uncategorized_selectors = [self.selectorDict[i] for i in range(self.num_categories)]
        self.categorized_selectors = [self.selectorDict[i] for i in range(self.num_categories, 2 * self.num_categories)]

        self.documentDict = {i: (0, documents[i]) for i in range(len(documents))}
        for docId in self.documentDict:
            self.selectorDict[0].addDoc(docId, self.documentDict[docId][1])

        def onMoveInBase(targetSelectorId):
            # move the selected item into category i
            docId = self.getSelectedDocId()
            if docId is not None:
                self.moveDocument(docId, targetSelectorId)
        def onMoveOut():
            docId = self.getSelectedDocId()
            if docId is not None:
                targetSelectorId = self.documentDict[docId][0] - self.num_categories
                if targetSelectorId >= 0:
                    self.moveDocument(docId, targetSelectorId)

        for i in range(self.num_categories):
            button = QPushButton("move in", self)
            onMoveIn = functools.partial(onMoveInBase, self.categorized_selectors[i].getId())
            button.clicked.connect(onMoveIn)
            self.grid.addWidget(button, 3, i, 1, 1)
        self.moveOutButton = QPushButton("move out", self)
        self.moveOutButton.clicked.connect(onMoveOut)

        self.setLayout(self.grid)
        self.grid.addWidget(self.moveOutButton, 4, 0, 1, self.num_categories)
        self.grid.addWidget(self.retrain_button, 5, 0, 1, self.num_categories - 1)
        self.grid.addWidget(self.save_button, 5, self.num_categories - 1, 1, 1)
        self.grid.addWidget(self.label, 6, 0, 1, self.num_categories)

        for docId in self.documentDict:
            doc = self.documentDict[docId][1]
            isTrain = doc.isTrain()
            if isTrain:
                label = doc.getLabel()
                self.moveDocument(docId, self.num_categories + label)

    def isCategorizedSelector(self, selectorId):
        return selectorId >= self.num_categories

    def isUncategorizedSelector(self, selectorId):
        return selectorId < self.num_categories

    def getSelectedDocId(self):
        for selectorId in self.selectorDict.keys():
            selector = self.selectorDict[selectorId]
            if selector.isAnythingSelected():
                return selector.getSelectedDocId()
        return None

    def onSelectorIndexClicked(self, selector, index, item):
        cur_id = selector.getId()
        cur_doc_id = selector.getSelectedDocId()
        doc = self.documentDict[cur_doc_id][1]
        self.label.setText(f'{doc.getTitle()}\n{doc.getBody()}')

        for selectorId in self.selectorDict.keys():
            selector = self.selectorDict[selectorId]
            if selectorId != cur_id:
                selector.blockSignals(True)
                selector.setCurrentIndex(-1)
                selector.blockSignals(False)

    def saveResults(self):
        documents = []
        classified_results = []
        # collect training emails and other emails
        for docId in range(len(self.documentDict)):
            selectorId, doc = self.documentDict[docId]
            documents.append(doc)
            classified_results.append(selectorId)
        Document.saveDocumentsTitlesAsCSV(documents, classified_results, './datasets/sort_results.csv')

    def retrainAndPredict(self):
        all_email_content = []
        train_email_content = []
        train_email_categories = []

        # collect training emails and other emails
        for docId in range(len(self.documentDict)):
            selectorId, doc = self.documentDict[docId]
            isManuallyCategorized = self.isCategorizedSelector(selectorId)
            content = doc.getBody()
            all_email_content.append(content)
            if isManuallyCategorized:
                train_email_content.append(content)
                categorized_label = selectorId - self.num_categories
                train_email_categories.append(categorized_label)

        model = TopicModel(train_email_content, train_email_categories, self.num_categories)
        print('model initialized')
        pred_all_email_categories = model.predict(all_email_content)

        for docId in range(len(self.documentDict)):
            selectorId, doc = self.documentDict[docId]
            pred_category = pred_all_email_categories[docId]
            if self.isUncategorizedSelector(selectorId):
                targetSelectorId = self.uncategorized_selectors[pred_category].getId()
                self.moveDocument(docId, targetSelectorId)
            else:
                correctSelectorId = self.categorized_selectors[pred_category].getId()
                if correctSelectorId != selectorId:
                    color = QColor('gray')
                else:
                    color = QColor('white')
                self.selectorDict[selectorId].setDocColor(docId, color)

    def moveDocument(self, document_id, to_id):
        from_id = self.documentDict[document_id][0]
        if from_id != to_id:
            doc = self.documentDict[document_id][1]
            self.documentDict[document_id] = (to_id, doc)

            from_selector = self.selectorDict[from_id]
            from_selector.removeDoc(document_id)
            to_selector = self.selectorDict[to_id]
            to_selector.addDoc(document_id, doc)

