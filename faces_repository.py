from cv2 import cv2
import numpy as np
import pathlib


def get_pictures_data(num_classes: int = 10, num_images: int = 10) -> tuple[np.ndarray, np.ndarray]:
    data_faces = []
    data_target = []
    data_folder = str(pathlib.Path(__file__).parent.resolve()) + "/pictures/s"

    for i in range(1, num_classes + 1):
        for j in range(1, num_images + 1):
            image = cv2.cvtColor(cv2.imread(data_folder + str(i) + "/" + str(j) + ".jpeg"), cv2.COLOR_BGR2GRAY)
            data_faces.append(image / 255)
            data_target.append(i)

    return np.array(data_faces), np.array(data_target)


NUM_FACES_OF_PERSON_IN_DATASET = 10


def split_data(train_data: np.ndarray, target_data: np.ndarray, num_faces_for_train: int) -> tuple[
    list, list, list, list]:
    all_faces_train_chunks = np.array_split(train_data, len(train_data) / NUM_FACES_OF_PERSON_IN_DATASET)
    all_faces_target_chunks = np.array_split(target_data, len(train_data) / NUM_FACES_OF_PERSON_IN_DATASET)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for person_images, person_targets in zip(all_faces_train_chunks, all_faces_target_chunks):
        np.random.shuffle(person_images)

        x_train.extend(person_images[:num_faces_for_train])
        y_train.extend(person_targets[:num_faces_for_train])

        x_test.extend(person_images[num_faces_for_train:])
        y_test.extend(person_targets[num_faces_for_train:])

    return x_train, y_train, x_test, y_test


def split_data_for_cross_validation(train_data: np.ndarray, target_data: np.ndarray, num_folds: int) -> tuple[
    list, list, list, list]:

    all_faces_train_chunks = np.array_split(train_data, len(train_data) / NUM_FACES_OF_PERSON_IN_DATASET)

    for chunk in all_faces_train_chunks:
        np.random.shuffle(chunk)

    all_faces_target_chunks = np.array_split(target_data, len(train_data) / NUM_FACES_OF_PERSON_IN_DATASET)

    faces_indexes = np.arange(NUM_FACES_OF_PERSON_IN_DATASET)

    np.random.shuffle(faces_indexes)

    split_faces_indexes = np.array_split(faces_indexes, num_folds)

    for test_indexes in split_faces_indexes:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train_indexes = set(faces_indexes) - set(test_indexes)

        for person_images, person_targets in zip(all_faces_train_chunks, all_faces_target_chunks):

            x_train.extend(person_images[index] for index in train_indexes)
            y_train.extend(person_targets[index] for index in train_indexes)

            x_test.extend(person_images[index] for index in test_indexes)
            y_test.extend(person_targets[index] for index in test_indexes)

        yield x_train, y_train, x_test, y_test


def zig_zag_transform(block):
    zigzag = []

    for index in range(1, len(block) + 1):
        slice = [i[:index] for i in block[:index]]

        diag = [slice[i][len(slice)-i-1] for i in range(len(slice))]

        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    for index in reversed(range(1, len(block))):
        slice = [i[:index] for i in block[:index]]

        diag = [block[len(block) - index + i][len(block) - i - 1]
                for i in range(len(slice))]

        if len(diag) % 2:
            diag.reverse()
        zigzag += diag

    return zigzag


