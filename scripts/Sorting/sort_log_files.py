import os
import re
import shutil

# path to files
#path = "C:\\Users\\robin\\Documents\\Uni\\MA\\models_15miosteps\\models"
#path= "C:\\Users\\robin\Documents\\Uni\MA\\remaining\\logs"
path = "C:\\Users\\robin\Documents\\Uni\MA\\finally_right_env\models"



def sort_from_server(path):
    # iterate over all files in path
    for file in os.listdir(path):

        # only consider files that match the pattern "PPO_Length_i_level_j_try_k"
        if re.match(r'PPO_length_(\d+)_level_(\d+\.\d+)_try_(\d+)', file):
            # extract length from filename
            length_match = re.search(r'PPO_length_(\d+)_level_(\d+\.\d+)', file)

            if length_match:
                length = length_match.group(1)
                level = length_match.group(2)

                # create folder if not exist
                length_path = os.path.join(path, f"Length_{length}_level_{level}")
                if not os.path.exists(length_path):
                    os.makedirs(length_path)

                # move files in folder
                shutil.move(os.path.join(path, file), os.path.join(length_path, file))
                print(f"Datei {file} was moved to {length_path}")
            else:
                print(f"File {file} does not correspond to the expected pattern")

def renaming_numbers(path):
    for file in os.listdir(path):


        match = re.match(r'PPO_length_(\d+)_level_(\d+\.\d+)_try_(\d+)', file)
        if match:
            # Neue Dateiname gemäß dem gewünschten Format erstellen
            new_filename = f"PPO_length_{match.group(1)}_level_{match.group(2)}_try_{int(match.group(3))+4}"
            os.rename(os.path.join(path, file), os.path.join(path, new_filename))
            file = new_filename



def delete_action_size(path):
    for file in os.listdir(path):
        match= re.match(r'PPO_length_(\d+)_level_(\d+\.\d+)_action_size_(\d+\.\d+)_try_(\d+)', file)
        if match:
            action_match=re.search(r'PPO_length_(\d+)_level_(\d+\.\d+)_action_size_(\d+\.\d+)_try_(\d+)', file)
            if action_match:
                action= action_match.group(3)
                length_path = os.path.join(path, f"action_size_{action}")
                if not os.path.exists(length_path):
                    os.makedirs(length_path)

                shutil.move(os.path.join(path, file), os.path.join(length_path, file))

        new_filename= f'PPO_length_{match.group(1)}_level_{match.group(2)}_try_{match.group(4)}'
        os.rename(os.path.join(path, file), os.path.join(path, new_filename))


#renaming_numbers(path)
#sort_from_server(path)