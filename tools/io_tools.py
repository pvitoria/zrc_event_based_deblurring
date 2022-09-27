import os


def find_files_by_template(folder, template):
    return [folder + '/' + file for file in sorted(os.listdir(folder)) if file.endswith(template)]
