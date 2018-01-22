import csv, os, shutil

def move_image(file_name, folder_name):
	shutil.move(file_name, folder_name + '/' + file_name)

with open('Data_Entry_2017.csv', 'rb') as csv_file:
    reader = csv.reader(csv_file, delimiter = ',')

    next(reader, None)

    for row in reader:
        if 'Cardiomegaly' in row[1]:
            move_image(row[0], 'cardiomegaly')
        elif 'No Finding' == row[1]:
            move_image(row[0], 'no finding')
        else:
            os.remove(row[0])

