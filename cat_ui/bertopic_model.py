import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression


MIN_DATASET_SIZE = 50


def draw_confusion_matrix_of_dataset(pred_labels, content_labels, ncategories):
    cm = confusion_matrix(content_labels, pred_labels)
    plt.imshow(cm, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.colorbar()
    label_range = np.arange(ncategories)
    label_strs = [str(label) for label in label_range]
    plt.xticks(label_range, label_strs)
    plt.yticks(label_range, label_strs)
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='green')
    plt.show()


class TopicModel:
    def __init__(self, train_email_content, train_email_categories, ncategories):
        if type(train_email_content[0]) != str:
            print(f'model can only accept string input, but got {type(train_email_content[0])}')
        if len(train_email_content) == 0:
            print('ERROR: Email content list is empty!')
        if len(train_email_categories) == 0:
            print('ERROR: Email category list is empty!')
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        empty_dimensionality_model = BaseDimensionalityReduction()
        clf = LogisticRegression()
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=empty_dimensionality_model,
            hdbscan_model=clf,
            ctfidf_model=ctfidf_model
        )

        # some bugs may happen if the dataset is too small, so we need to pad it somehow
        # see https://github.com/MaartenGr/BERTopic/issues/97
        if len(train_email_content) < MIN_DATASET_SIZE:
            dup_count = int(MIN_DATASET_SIZE / len(train_email_content)) + 1
            train_email_content = train_email_content * dup_count
            train_email_categories = train_email_categories * dup_count

        topics, probs = self.model.fit_transform(train_email_content, y=train_email_categories)

        map_matrix = np.zeros((ncategories, ncategories), dtype=np.int32)
        for i in range(len(topics)):
            if topics[i] >= 0:  # put an if here because it may be -1? not sure
                map_matrix[train_email_categories[i], topics[i]] += 1
        top_category = np.argmax(map_matrix, axis=0)
        self.cluster_to_category = [top_category[i].item() for i in range(len(top_category))]

        # self.model.save("my_model", serialization="safetensors")
        self.ncategories = ncategories

    def predict_single(self, email_content):
        if type(email_content) != str:
            print(f'model can only accept string input, but got {type(email_content)}')
        categories, probs = self.model.transform(email_content)
        prediction = self.cluster_to_category[categories[0]]
        return prediction

    def predict(self, test_email_content):
        results = self.model.transform(test_email_content)
        categories = results[0]
        predictions = [self.cluster_to_category[categories[i]] for i in range(len(categories))]
        return predictions

    def eval_on_test_set(self, test_email_content, test_email_label, show_graph=False):
        predict_label = self.predict(test_email_content)
        acc = np.average(
            np.array([test_email_label[i] == predict_label[i] for i in range(len(predict_label))],
                     dtype=np.int32))
        print(f'accuracy: {acc}')
        if show_graph:
            draw_confusion_matrix_of_dataset(test_email_label, predict_label, self.ncategories)
