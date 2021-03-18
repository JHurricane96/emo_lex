import csv

def read_csv_file(file_path):
    with open(file_path, encoding="utf-8", newline='') as file:
        file_reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        yield from file_reader

def write_all_rows(output_file_path, rows, mode='w'):
    with open(output_file_path, mode, encoding="utf-8", newline='') as output_file:
        output_file_writer = csv.writer(output_file, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
        output_file_writer.writerows(rows)
