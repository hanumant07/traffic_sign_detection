from collections import defaultdict
import csv
import os
import numpy as np

_WRITEUP_DIR = './writeup_dir'


def save_fig_for_writeup(fig, fname):
  """Function to save a matplot lib figure as image file.

  Args:
    fig: matplot lib fig instance.
    fname: base file name of image file.

  Raises:
    ValueError: if figname or fname are not valid entries.
  """

  if fig is None:
    raise ValueError('Pass valid matplot lib fig')
  if fname is None:
    raise ValueError('Pass valid filename')
  file_path = os.path.join(_WRITEUP_DIR, fname)
  fig.savefig(file_path)


class TrafficDataInfo(object):
  """Parse and analyze the traffic sign data."""

  def _populate_ids_to_names(self, names_files):
    """Parse csv file containing class id to sign names.

    Args:
      names_files: csv file name.

    Raises:
      ValueError if names_files is not a str
    """
    if not isinstance(names_files, str):
      raise ValueError('names_files needs to be valid string')
    with open('signnames.csv', mode='r') as csv_file:
      signs = csv.DictReader(csv_file)
      for row in signs:
        self._ids_to_names[int(row['ClassId'])] = row['SignName']

  def __init__(self, names_file):
    """Inits TrafficDataInfo.

    Args:
      names_file: Csv file that contains a mapping to traffic sign names
    """
    self._ids_to_names = {}
    self._populate_ids_to_names(names_file)

  def organize_data_by_label(self, labels):
    """Create a mapping from class id to sample index from dataset.

    Args:
      labels: A list of class ids. It is assumed entries in the list correspond
        to same position of the corresponding sample in the dataset.

    Returns:
      A dictionray that maps label id to sample index.
    """
    label_to_data = defaultdict(list)
    for i in range(len(labels)):
      label_to_data[labels[i]].append(i)
    return label_to_data

  def get_name_for_label(self, label_id):
    """Map label id to sign name.

    Args:
      label_id: The id that represents the sign as per the input csv file.

    Returns:
      sign name for given id.
    """
    return self._ids_to_names[label_id]

  def get_hist_info(self, labels):
    """Get histogram information for the datasamples.

    The histogram information pertains to the number of samples for a given
    class.

    Args:
      labels: The labels is a list of class ids corresponding to the samples in
        the data set.

    Returns:
      sorted list of class ids and their respective sample counts.
    """
    return np.unique(labels, return_counts=True)

  def add_existing_samples(self, training_x, training_y, new_x, new_y):
    """Add new samples to the existing training set.

    Args:
      training_x: Existing training set.
      training_y: Existing training labels.
      new_x: New samples to add.
      new_y: Corresponding new labels to add.

    Returns:
      new training samples.
    """
    labels, label_counts = self.get_hist_info(training_y)
    new_data_by_label = self.organize_data_by_label(new_y)
    max_count = label_counts.max()

    for i, label in enumerate(labels):
      if label_counts[i] > (max_count / 2):
        continue
      valid_indices = new_data_by_label[label]
      new_samples = new_x[np.array(valid_indices)]
      new_labels = np.repeat(label, len(new_samples))
      training_x = np.concatenate((training_x, new_samples), axis=0)
      training_y = np.concatenate((training_y, new_labels), axis=0)
    return training_x, training_y
