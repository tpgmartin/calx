import scipy as sp
from sklearn import linear_model

# Example usage: binary classification
# 1. Initialise LimeTextExplainer with class names
# 2. Create explaination instance from example document, with 
#    explainer.explain_instance() with params
#    instance to be explained, classifier function, num_features to use
# 3. Generate list of features for explanations with as_list() on 
#    explanation object
# 4. Print explanation as pyplot with exp.as_pyplot_figure()
# 5. Other explanation methods: show_in_notebook, save_to_file, 
#    show_in_notebook


class Calx(object):

    # lime base methods required explain_instance_with_data()

    # other params, kernel_width, verbose
    def __init__(self):
        # param class names which are target labels
        # self.base = ... base object ...
        # kernel = ...
        self.kernel_fn = lambda d: np.sqrt(np.exp(-(d**2) / 25 ** 2))
        # don't need class_names as generate in explain_instance

    def explain_instance(self, instance, classifier_fn, num_features=10):
        # returns explanation object with explanations
        # 1. Create sparse matrix from instance param
        instance = sp.sparse.csr_matrix(instance)
        # 2. Call data_labels_distances_mapping
        data, yss, distances, mapping = self.__data_labels_distances_mapping(instance, classifier_fn, num_samples)
        # 3. Create class_names if not specified
        # 4. Create explanation object
        # 5. Call predict_proba
        # 6. map_exp
        # 7. Iterate over labels and generate explanations
        # 8. return explantion object with explanations
    
    def __data_labels_distances_mapping(self, instance, classifier_fn, num_samples):
        pass
    
    def explain_instance_with_data():
        pass
    
    @staticmethod
    def generate_lars_path():
        pass
    
    # minimum function required for output
    # def as_list()
