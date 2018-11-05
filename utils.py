import csv
import os
from collections import defaultdict

_writeup_dir = './writeup_dir'

'''
save_fig_for_writeup: Function to save a matplot lib figure as image file

    @fig : matplot lib fig
    @ fname : base file name of image file
'''
def save_fig_for_writeup(fig, fname):
    assert(fig is not None)
    file_path = os.path.join(_writeup_dir, fname)
    fig.savefig(file_path)


'''
TrafficDataInfo: Class to parse and analyze the traffic sign data
'''
class TrafficDataInfo():
    
    def _populate_ids_to_names(self, names_files):
        with open('signnames.csv', mode='r') as csv_file:
            signs = csv.DictReader(csv_file)
            for row in signs:
                self._ids_to_names[int(row['ClassId'])] = row['SignName']

    '''
    @names_file: Csv file that contains a mapping to traffic sign names
    '''
    def __init__(self, names_file):
        self._ids_to_names = {}
        self._populate_ids_to_names(names_file)


    '''
    organize_data_by_label: Create a mapping from class id to sample index from dataset
    @labels: A list of class ids. It is assumed entries in the list correspond to same position
             of the corresponding sample in the dataset
    '''
    def organize_data_by_label(self, labels):
        label_to_data = defaultdict(list)
        for i in range(len(labels)):
            label_to_data[labels[i]].append(i)
        return label_to_data


    '''
    get_name_for_label: Map label id to sign name
    @label_id: The id that represents the sign as per the input csv file
    '''
    def get_name_for_label(self, label_id):
        return self._ids_to_names[label_id]


    '''
    get_hist_info: Get histogram information for the datasamples
    @labels: The labels is a list of class ids corresponding to the samples in the data set.
    The histogram information pertains to the number of samples for a given class
    '''
    def get_hist_info(self, labels):
        return np.unique(labels, return_counts=True)


    '''
    add_existing_samples: Add new samples to the existing training set
    @training_x: Existing training set
    @training_y: Existing training labels
    @new_x: New samples to add
    @new_y: Corresponding new labels to add
    '''
    def add_existing_samples(self, training_x, training_y, new_x, new_y):
        labels, label_counts = self.get_hist_info(training_y)
        new_data_by_label = self.organize_data_by_label(new_y)
        max_count = label_counts.max()

        for i, label in enumerate(labels):
            if (label_counts[i] > (max_count / 2)):
                continue
            valid_indices = new_data_by_label[label]
            new_samples = new_x[np.array(valid_indices)]
            new_labels = np.repeat(label, len(new_samples))
            training_x = np.concatenate((training_x, new_samples), axis=0)
            training_y = np.concatenate((training_y, new_labels), axis=0)
        return training_x, training_y