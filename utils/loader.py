import os

from Dataset.dataset import DataLoaderTrain2, DataLoaderVal, DataLoaderTest, DataLoaderTestSR, DataLoaderTrainNoise, DataLoaderValNoise, DataLoaderTrainDPD,DataLoaderTrainGoPro
def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain2(rgb_dir, img_options, None)

def get_training_data2(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainNoise(rgb_dir, img_options, None)

def get_training_data3(rgb_dir, patchsize):
    assert os.path.exists(rgb_dir)
    return [DataLoaderTrainDPD(rgb_dir, p, None) for p in patchsize]

def get_training_data4(rgb_dir, patchsize):
    assert os.path.exists(rgb_dir)
    return [DataLoaderTrainGoPro(rgb_dir, p, None) for p in patchsize]

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)

def get_validation_data2(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderValNoise(rgb_dir, None)

def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)


def get_test_data_SR(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestSR(rgb_dir, None)