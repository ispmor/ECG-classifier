import visualise_examples

if __name__ == '__main__':
    all_records = []
    headers = []

    #data_sorting.sort_signals("./PhysioNetChallenge2020_Training_CPSC/Training_WFDB/")
    '''for i in range(1, 2):
        name = "A" + str(i).zfill(4)
        x = wfdb.io.rdsamp("./PhysioNetChallenge2020_Training_CPSC/Training_WFDB/" + name)
        all_records.append(x[0])
        headers.append(x[1])

    for record in headers:
        print(record)'''
    visualise_examples.plot_each_example()
