import mne
from pathlib import Path
import pandas as pd

root = Path("/media/mohammad/NewVolume/signalDataset/haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings")   # مسیر پوشه





def read_data_haaglanden(
     root:str,
     number_persion : int = 1,
     print_data_analez : bool = False,
     stage_map = {
                    "Sleep stage W": 0, "W": 0,
                    "Sleep stage N1": 1, "Sleep stage 1": 1, "N1": 1,
                    "Sleep stage N2": 2, "Sleep stage 2": 2, "N2": 2,
                    "Sleep stage N3": 3, "Sleep stage 3": 3, "Sleep stage 4": 3, "N3": 3,
                    "Sleep stage R": 4, "R": 4, "REM": 4,
                    }
     ):

     psg_file = root / f"SN{number_persion:03d}.edf" # type: ignore

     raw = mne.io.read_raw_edf(psg_file, preload=True, stim_channel=None, verbose=False)

     data_x = raw.get_data()
     channel_name = raw.ch_names





     scoring_edf = root / f"SN{number_persion:03d}_sleepscoring.edf" # type: ignore

     # خواندن annotation ها
     ann = mne.read_annotations(scoring_edf)

     # تبدیل به DataFrame
     df = pd.DataFrame({
     "start_sec": ann.onset,
     "duration_sec": ann.duration,
     "label": ann.description
     })



     valid_labels = set(stage_map.keys())
     df_clean = df[df["label"].isin(valid_labels)].reset_index(drop=True)

     df_clean["label"] = df_clean["label"].map(stage_map)

     data_y = df_clean.astype(int).to_numpy()


     if print_data_analez:
          print('-' * 50)
          print('x analize : ')
          print('shape x: ',data_x.shape) # type: ignore
          print('channel signal :',channel_name)
          print('-' * 50)
          print('y analize :')
          print('number vlaue :' ,df["label"].value_counts())
          print('number vlaue (clean label) :' ,df_clean["label"].value_counts())
          print('shape label:', data_y.shape)
          print('-' * 50)

     
     return data_x, data_y
    




from tqdm import tqdm
def create_np_haaglanden(
    root:str,
    number_all_persion : int ,
    #dir_save_data : str,
    print_data_analez : bool = False,
    stage_map = {
                "Sleep stage W": 0, "W": 0,
                "Sleep stage N1": 1, "Sleep stage 1": 1, "N1": 1,
                "Sleep stage N2": 2, "Sleep stage 2": 2, "N2": 2,
                "Sleep stage N3": 3, "Sleep stage 3": 3, "Sleep stage 4": 3, "N3": 3,
                "Sleep stage R": 4, "R": 4, "REM": 4,
                }
    ):
    for i in tqdm(range(number_all_persion)):
        number_persion = i + 1
        try:
            x, y = read_data_haaglanden(
            root = root,
            number_persion  = number_persion ,
            print_data_analez  = print_data_analez,
            stage_map = stage_map
            )
            
        except:
            print(f'file {number_persion} is not define.')
