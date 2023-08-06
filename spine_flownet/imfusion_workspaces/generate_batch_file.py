import fnmatch
import os


def recursive_search(directory, pattern):
    matches = []
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.normpath(os.path.join(root, filename)))

    return matches


def generate_entry_dict(input_matches):

    replace_value_list = ["21 22 23 24", "20 22 23 24", "20 21 23 24", "20 21 22 24", "20 21 22 23"]

    entry_dict = {'INPUTFILE': [], 'OLDVALUES': [], 'OUTPUTSTL': [], 'OUTPUTPC': [] }
    for input_file in input_matches:

        out_folder = os.path.split(input_file)[0]

        for i, replace_value in enumerate(replace_value_list):
            out_stl = os.path.join(out_folder, "v" + str(i + 1) + ".stl")
            out_pc = os.path.join(out_folder, "v" + str(i + 1) + ".txt")

            entry_dict['INPUTFILE'].append(input_file)
            entry_dict['OLDVALUES'].append(replace_value)
            entry_dict['OUTPUTSTL'].append(out_stl)
            entry_dict['OUTPUTPC'].append(out_pc)

    return entry_dict

def main(batch_file_path, data_path):
    matches = recursive_search(data_path, "*seg.nii.gz")
    entry_dict = generate_entry_dict(matches)

    with open(batch_file_path, "w") as fid:
        fid.write('INPUTFILE;OLDVALUES;OUTPUTSTL;OUTPUTPC')
        for i in range(len(entry_dict['INPUTFILE'])):
            file_string = "\n" + entry_dict['INPUTFILE'][i] + ";" + \
                          entry_dict['OLDVALUES'][i] + ";" + \
                          entry_dict['OUTPUTSTL'][i] + ";" + \
                          entry_dict['OUTPUTPC'][i]

            fid.write(file_string)

if __name__ == "__main__":
    data_path = "E:/NAS/jane_project/new_spines_Jane/"
    batch_file_path = "E:/NAS/jane_project/new_spines_Jane/batch.txt"
    main(batch_file_path, data_path)
