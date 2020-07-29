import wfdb
import os
import shutil
import numpy as np

def sort_signals(data_path):
    absolute_data_path = os.path.abspath(os.getcwd() + data_path[1:])
    for i in range(1, 6877):
        file_name = "A" + str(i).zfill(4)
        file = wfdb.io.rdsamp(data_path + file_name)

        header = file[1]
        signal_class = header["comments"][2][4:]

        class_dir_path = os.path.abspath(os.getcwd()) + "/data/" + signal_class
        if not os.path.exists(class_dir_path):
            os.mkdir(class_dir_path)

        if not os.path.exists(class_dir_path + "/" + file_name + ".mat"):
            shutil.copy(absolute_data_path + "/" + file_name + ".mat", class_dir_path)
            shutil.copy(absolute_data_path + "/" + file_name + ".hea", class_dir_path)


def move_clean_test_signals():
    test_dir = os.getcwd() + "/only-clean-test/"
    data_dir = os.getcwd() + "/only-clean/"
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for folder_name in dirnames:
            name = folder_name.split("/")[-1]
            actual_class_dir = data_dir + name
            print(actual_class_dir)
            f = []
            for (dirpath, dirnames, filenames) in os.walk(actual_class_dir):
                single_names = [n for n in filenames if ".hea" not in n]
                f.extend(single_names)
                break

            print(f)
            np.random.shuffle(f)
            test_size = int(len(f) * 0.8)
            f_test = f[test_size:]
            class_dir_path = os.path.abspath(os.getcwd()) + "/only-clean-test/" + name
            absolute_data_path = os.path.abspath(os.getcwd()) + "/only-clean/" + name

            print(class_dir_path + "------------" + absolute_data_path)
            if not os.path.exists(class_dir_path):
                os.mkdir(class_dir_path)

            for f_name in f_test:
                file_name = f_name[:-4]
                if not os.path.exists(class_dir_path + "/" + file_name + ".mat"):
                    shutil.move(absolute_data_path + "/" + file_name + ".mat", class_dir_path)
                    shutil.move(absolute_data_path + "/" + file_name + ".hea", class_dir_path)


if __name__ == "__main__":
    move_clean_test_signals()
