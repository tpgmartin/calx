from calx.explanation import Explanation
import numpy as np
import scipy as sp
import sklearn

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
    # assume class names are supplied
    def __init__(self, class_names, kernel_width=25, verbose=False):
        self.kernel_fn = lambda d: np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
        self.class_names = class_names
        self.verbose = verbose
        # self.base = lime_base # also need to have methods - don't need if 
        # not specifying options in method explain instance
        # generate_lars_path, explain_instance_with_data

    # reference: explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
    def explain_instance(self, instance, classifier_fn, labels=(1), num_features=10, num_samples=5000):
        # returns explanation object with explanations
        # 1. Create sparse matrix from instance param
        instance = sp.sparse.csr_matrix(instance)
        # 2. Call data_labels_distances_mapping
        data, yss, distances, mapping = self.__data_labels_distances_mapping(
                                        instance, classifier_fn, num_samples)
        # 3. Create class_names if not specified - ignore this step
        # 4. Create explanation object
        ret_exp = Explanation(class_names=self.class_names)
        # 5. Call predict_proba
        ret_exp.predict_proba = yss[0]
        # 6. map_exp
        map_exp = lambda exp: [(mapping[x[0]], x[1]) for x in exp]
        # 7. Iterate over labels and generate explanations
        # skip for now
        # 8. return explantion object with explanations
        return ret_exp
    
    @staticmethod
    # Generates a neighbourhood around a prediction.
    def __data_labels_distances_mapping(instance, classifier_fn, num_samples):
        features = instance.nonzero()[1]

        # data
        vals = np.array(instance[instance.nonzero()])[0]
        doc_size = len(sp.sparse.find(instance)[2])
        sample = np.random.randint(1, doc_size, num_samples - 1)
        data = np.zeros((num_samples, len(features)))
        inverse_data = np.zeros((num_samples, len(features)))
        data[0] = np.ones(doc_size)
        inverse_data[0] = vals
        features_range = range(len(features))
        for i, size in enumerate(sample, start=1):
            active = np.random.choice(features_range, size, replace=False)
            data[i, active] = 1
            inverse_data[i, active] = vals[active]

        sparse_inverse = sp.spacing.lil_matrix((inverse_data.shape[0], 
                                                instance.shape[1]))
        sparse_inverse[:, features] = inverse_data
        sparse_inverse = sp.sparse.csr_matrix(sparse_inverse)

        # labels
        labels = classifier_fn(sparse_inverse)

        # distances
        distance_fn = lambda x: sklearn.metrics.pairwise.cosine_distnace(x[0], x)[0] * 100
        distances = distance_fn(sparse_inverse)

        # mapping
        mapping = features

        return data, labels, distances, mapping
