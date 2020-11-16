import wfdb
import os
import shutil
import numpy as np

def sort_signals(data_path):
    total = 6878
    for i in range(1, total):
        file_name = "A" + str(i).zfill(4)
        if os.path.exists(data_path + file_name + ".mat"):
            file = wfdb.io.rdsamp(data_path + file_name)

            header = file[1]
            signal_class = header["comments"][2][4:]
            print("Moving file: %s\t to: %s\t progress: %d/%d" %(file_name, signal_class, i, total), end="\r")

        
            class_dir_path = "/home/puszkar/ecg/data/" + signal_class
            if not os.path.exists(class_dir_path):
                os.mkdir(class_dir_path)

            if not os.path.exists(class_dir_path + "/" + file_name + ".mat"):
                shutil.move(data_path + file_name + ".mat", class_dir_path)
                shutil.move(data_path + file_name + ".hea", class_dir_path)
        


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



def get_clean_diagnosis_train_test_split():
    data_dir = os.path.dirname(os.getcwd()) + "/data/"
    training_dir = os.path.dirname(os.getcwd()) + "/data/training/"
    test_dir = os.path.dirname(os.getcwd()) + "/data/test/"
    
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    

    d = [x[0] for x in os.walk(data_dir)]
    dirs = []
    dirs_path = []
    for directory_name in d:
        last_part = directory_name.split("/")[-1]
        if all(x not in last_part for x in [",", "training", "test"]):
            dirs_path.append(directory_name)
            dirs.append(last_part)
    dirs = dirs[1:]
    dirs_path = dirs_path[1:]
    print(dirs_path)

    dataset = []
    for folder_name in dirs_path:
        for (_, dirs, files) in os.walk(folder_name):
            dataset = dataset + [(folder_name + "/" + f[:-4], f[:-4]) for f in files if ".mat" not in f]
            
    np.random.shuffle(dataset)
    train_size = int(len(dataset) * 0.8)
    test_dataset = dataset[train_size:]
    train_dataset = dataset[:train_size]
    
    print("#" * 100)
    print(test_dataset[0:10])
    print("#" * 100)
    print(train_dataset[0:10])
    print(f"dataset len: {len(dataset)},\t trainining_dataset len: {len(train_dataset)},\t test_dataset len: {len(test_dataset)}" )
    
    for f_path, f_name in test_dataset:
        if not os.path.exists(test_dir + f_name + ".mat"):
            shutil.copy(f_path + ".mat", test_dir)
            shutil.copy(f_path + ".hea", test_dir)
            
    for f_path, f_name in train_dataset:
        if not os.path.exists(training_dir + f_name + ".mat"):
            shutil.copy(f_path + ".mat", training_dir)
            shutil.copy(f_path + ".hea", training_dir)
     
    

def get_classes_test_split(classes_array):
    data_dir = os.path.dirname(os.getcwd()) + "/ecg/data/"
    if not os.path.exists(data_dir):
        raise ValueError(f"Data_Dir is not properly set, try calling script from different place. Current data_dir: {data_dir}")
    output_dir = data_dir + "-".join(classes_array) + "test"
    if not os.path.exists(output_dir):
        print(f"Output_Dir does not exist, creating: {output_dir}")
        os.mkdir(output_dir)
            
    dataset = []
    for folder_name in classes_array:
        for (_, dirs, files) in os.walk(data_dir + folder_name):
            dataset = dataset + [(data_dir + folder_name + "/" + f[:-4], f[:-4]) for f in files if ".mat" not in f]
            
    np.random.shuffle(dataset)
    train_size = int(len(dataset) * 0.8)
    test_dataset = dataset[train_size:]
    print(test_dataset)
    print(f"dataset len: {len(dataset)}, test_dataset len: {len(test_dataset)}" )

    for f_path, f_name in test_dataset:
        if not os.path.exists(output_dir + f_name + ".mat"):
            shutil.copy(f_path + ".mat", output_dir)
            shutil.copy(f_path + ".hea", output_dir)     



if __name__ == "__main__":
    #move_clean_test_signals()
    #sort_signals("/home/puszkar/ecg/data/")
    #get_clean_diagnosis_train_test_split()
    get_classes_test_split(['LBBB', 'Normal', 'RBBB', 'AF', 'STE', 'PAC', 'PVC'])
