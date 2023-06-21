import csv


def print_from_cpps():
    data_file_path = 'CPPS1.1-no-dup.csv'
    with open(data_file_path, 'r', newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        row_list = list(csv_reader)
        counter = {'health': 0}
        for each_label in row_list:
            seg, cate, cate_label, collect, share, store = each_label
            if cate == 'health':
                counter['health'] += 1
        print("number of health labels: ")
        print(counter)


print_from_cpps()