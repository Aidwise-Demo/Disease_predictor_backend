import os
import time

def remove_old_files():
    # Define the folders to clean up
    folders = ['uploads', 'uploadsCovid', 'input']

    # Get the current time
    now = time.time()

    # Define the age limit (1 week in seconds)
    age_limit = 7 * 24 * 60 * 60  # 7 days

    for folder in folders:
        # Check if the folder exists
        if os.path.exists(folder):
            # Iterate over all files in the folder
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                # Check if it's a file
                if os.path.isfile(file_path):
                    # Get the file's last modified time
                    file_mod_time = os.path.getmtime(file_path)
                    # Check if the file is older than the age limit
                    if now - file_mod_time > age_limit:
                        os.remove(file_path)
                        print(f'Removed: {file_path}')
        else:
            print(f'Folder not found: {folder}')

# Optional: If you want to run the function when executing this file directly
if __name__ == "__main__":
    remove_old_files()
