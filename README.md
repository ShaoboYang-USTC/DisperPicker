# DisperPicker

by Shaobo Yang, University of Science and Technology of China, 2020

E-mail: <yang0123@mail.ustc.edu.cn>

This tool mainly targets at automatically extracting dispersion curves via deep learning for the image transformation technique (EGFAnalysisTimeFreq, https://github.com/ShaoboYang-USTC/EGFAnalysisTimeFreq) and the associated software developed by Huajian Yao, which is widely used in the community for dense array ambient noise analysis.

References: Yang, S., Zhang, H., Gu, N., Gao, J., Xu, J., Jin, J., Li, J., and Yao H. (2022). Automatically Extracting Surface Wave Group and Phase Velocity Dispersion Curves from Dispersion Spectrograms Using a Convolutional Neural Network. *Seismological Research Letters*, doi: https://doi.org/10.1785/0220210280.

## 1. Installation

* Platform: Linux
* Download repository
* Install Python 3.6
* Install dependencies: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

## 2. Applications

* Put the test data into `./data/TestData`. When saving the dispersion spectrograms, the velocity step (dv) must be 0.005 km/s and the recommended period step (dT) is 0.1 s. The dispersion spectrograms can be derived from EGFs or CFs using EGFAnalysisTimeFreq.

* Check the parameters in the configuration file: `./config/config.py`

* Write the station location information in `./config/station.txt`

* Run the detection program: `python pick_v.py` 

## 3. Picking results

* The picking results are saved in `./result/pick_result` and some figures are saved in `./result/plot1` and `./result/plot2`

## 4. Quality Control

* Run `./result/process/process_C.py` for QC of the phase velocity dispersion curves and `./result/process/process_G.py` for group velocity.
* The dispersion curves after QC is saved in `./result/process/new/` and all of the dispersion curves are shown in `DPN_C.jpg` and `DPN_G.jpg`.
* Run `./result/process/plot_res.py` to plot each dispersion spectrogram and the corresponding DisperPicker picked dispersion curves. The figures are save in `./result/process/new/plot/`.

