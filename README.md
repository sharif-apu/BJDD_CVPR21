# BJDD_CVPR21
This is the official implementation of paper title "A Two-stage Deep Network for High Dynamic Range Image Reconstruction" from CVPRW21. To download full paper **[[Click Here](https://competitions.codalab.org/competitions/28161#participate)]**.


**Please consider to cite this paper as follows:**
```
@inproceedings{sharif2021hdr,
  title={A Two-stage Deep Network for High Dynamic Range Image Reconstruction},
  author={Sharif, SMA and Naqvi, Rizwan Ali, Biswas, Mithun and Sungjun, Kim},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={},
  year={2021}
}
```

# Overview
<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/twostageHDR_NTIRE21/blob/master/images/overviewUp.png" alt="Overview"> </br>
</p>

**Figure:** Overview of the proposed method.  The proposed method comprises a two-stage deep network.  Stage-I aims toperform image enhancement task denoising, exposure correction, etc.  Stage-II of the proposed method intends to performtone mapping and bit-expansion.


# Comparison with state-of-the-art sigle-shot LDR to HDR Deep methods </br>

<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/twostageHDR_NTIRE21/blob/master/images/comp.png" alt="Overview"> </br>
</p>

**Figure:** Quantitative comparison between proposed method and existing learning-based single-shot LDR to HDR methods..

# Prerequisites
```
Python 3.8
CUDA 10.1 + CuDNN
pip
Virtual environment (optional)
```

# Installation
**Please consider using a virtual environment to continue the installation process.**
```
git clone https://github.com/sharif-apu/BJDD_CVPR21.git
cd BJDD_CVPR21
pip install -r requirement.txt
```

# Testing
** [[Click Here](https://drive.google.com/drive/folders/1_ziIMjK9vGg-P_7Wxit96bnfHiO4_wQw?usp=sharinge)]** to download pretrained weights and save it to weights/ directory for inferencing with Quad-bayer CFA</br>
```python main.py -i``` </br>

A few testing images are provided in a sub-directory under testingImages (i.e., testingImages/sampleImages/)</br>
In such occasion, reconstructed image(s) will be available in modelOutput/sampleImages/. </br>

**To inference with custom setting execute the following command:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages -ns=sigma(s)``` </br>
Here,**-ns** specifies the standard deviation of a Gaussian distribution (i.e., -ns=5, 10, 15),**-s** specifies the root directory of the source images
 (i.e., testingImages/), and **-d** specifies the destination root (i.e., modelOutput/).


# Training
To start training we need to sampling the images according to the CFA pattern and have to pair with coresponding ground-truth images.
To sample images for pair training please execute the following command:

```python main.py -ds -s /path/to/GTimages/ -d /path/to/saveSamples/ -g 2 -n 10000 ```
</br> Here **-s** flag defines your root directory of GT images, **-d** flag defines the directory where sampled images should be saved, and **-g** flag defines the binnig factr (i.e., 1 for bayer CFA, 2 for Quad-bayer), **-n** defines the number of images have to sample (optional)</br>


</br> After extracting samples, please execute the following commands to start training:

```python main.py -ts -e X -b Y```
To specify your trining images path, go to mainModule/config.json and update "gtPath" and "targetPath" entity. </br>You can specify the number of epoch with **-e** flag (i.e., -e 5) and number of images per batch with **-b** flag (i.e., -b 12).</br>


**For transfer learning execute:**</br>
```python main.py -tr -e -b ```


# Bayer Testing
We also trained our model with Bayer CFA. To download pretrained Bayer weights **[[Click Here](https://drive.google.com/drive/folders/125hFTHR5qpJy4AKhtjxFhZJ5aPxQI4TE?usp=sharing)]**. In such occasion, please update binning factor entity in mainModule/config.json file.


# Others
**Check model configuration:**</br>
```python main.py -ms``` </br>
**Create new configuration file:**</br>
```python main.py -c```</br>
**Update configuration file:**</br>
```python main.py -u```</br>
**Overfitting testing** </br>
```python main.py -to ```</br>

# Contact
For any further query, feel free to contact us through the following emails: sma.sharif.cse@ulab.edu.bd, rizwanali@sejong.ac.kr, or mithun.bishwash.cse@ulab.edu.bd
