import csv
import os
from FileLoop import get_file_paths
import video_process 

def ensure_csv_with_header(file_path, header):
    """Ensure a CSV file exists and has a header row."""
    if not os.path.exists(file_path):  # Check if the file exists
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
              # Write the header only if the file is newly created

def main():
    directory_path = input('Enter your directory path for the video folder: ')  # Replace with your folder path

    deviation_file = r".\data\deviation_data_new.csv"
    rate_change_file = r".\data\rate_change_data_new.csv"
    normalized_file = r".\data\normalized_data_new.csv"

    #"C:\\Users\\benha\\Documents\\CMPSC\\CMPSC 445\\SOBER_DATA"

#C:\Users\benha\Documents\CMPSC\CMPSC 445\Final Project\ML_DrinkingModel\data\normalized_data_new.csv
#C:\Users\benha\Documents\CMPSC\CMPSC 445\Final Project\ML_DrinkingModel\Scripts\DU(a)I.py

    # Ensure each file exists and has a header
    ensure_csv_with_header(deviation_file, header=None )
    ensure_csv_with_header(rate_change_file, header=None)
    ensure_csv_with_header(normalized_file, header=None)

    file_list = get_file_paths(directory_path)

    # Open files in append mode
    with open(deviation_file, mode='a', newline='') as dev_file, \
         open(rate_change_file, mode='a', newline='') as roc_file, \
         open(normalized_file, mode='a', newline='') as norm_file:

        dev_writer = csv.writer(dev_file)
        roc_writer = csv.writer(roc_file)
        norm_writer = csv.writer(norm_file)

        counter = 1

        for x in file_list:

            _, file_ext = os.path.splitext(x)

            # Check if the file has the desired extension (case-insensitive)
            if file_ext.lower() not in ('.mov', '.mp4'):
                continue

            # Process the video file
            deviationVal, rateOfChange, normalizedData = video_process.main(x, False, 2)

            # Write results to respective CSVs
            dev_writer.writerow(deviationVal)
            roc_writer.writerow(rateOfChange)
            norm_writer.writerow(normalizedData)

            print('####################################\n\nCOUNT ',
                  counter, '\n\n#################################')
            print(f"Processed {x}: Deviation={deviationVal}, RateOfChange={rateOfChange}, NormalizedData={normalizedData}")
            counter += 1


if __name__ == "__main__":
    main()