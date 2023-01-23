'''
    Testing utilities
'''

import csv


def save_sim_errors_to_file(filename: str, header: list, data):

    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


def save_results_to_file(filename: str, header: list, data):
    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)