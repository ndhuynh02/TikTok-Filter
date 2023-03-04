import os
import sys
sys.path.append(os.path.abspath("../scr"))

from src.data.dlip_datamodule import DLIPDataModule

if __name__== 'main':
    data = DLIPDataModule()
    data.prepare_data()

    assert len(data) == 7674