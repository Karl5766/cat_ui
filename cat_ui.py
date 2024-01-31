import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
import cat_ui.sorting_panel as sorting_panel
from cat_ui.sorting_panel import Document
import pandas as pd

ROOT = './datasets'

if __name__ == '__main__':
    app = QApplication(sys.argv)

    df_review = pd.read_excel(f'{ROOT}/Winter 2024 Scotia DSD Data Set.xlsx', sheet_name=None)['Sheet1']
    ids = df_review['Review_ID'].values
    reviews = df_review['Review'].values

    NUM_CATEGORIES = 3
    labels, is_train_arr = [], []
    with open('./datasets/sort_results_revised.csv', 'r') as infile:
        for line in infile.readlines():
            strs = line.split(',')
            i, label = int(strs[0]), int(strs[1])
            if label >= NUM_CATEGORIES:
                label -= NUM_CATEGORIES
                is_train_arr.append(True)
            else:
                is_train_arr.append(False)
            labels.append(label % NUM_CATEGORIES)
    with open('./datasets/hotel.csv', 'w') as outfile:
        strs = []
        strs.append('Review_ID, Is_Labeled, Label')
        for i in range(len(df_review)):
            label = labels[i]
            is_train = is_train_arr[i]
            if label == 0:
                label_str = 'BankApp'
            elif label == 1:
                label_str = 'Maybe'
            else:
                label_str = 'Hotel'
            strs.append(f'{i}, {is_train}, {label_str}')
        outfile.write('\n'.join(strs))
    documents = [Document(str(ids[i]), reviews[i], labels[i], is_train_arr[i]) for i in range(len(ids))]

    window = sorting_panel.SortingPanel(documents, NUM_CATEGORIES)
    window.show()
    sys.exit(app.exec_())
