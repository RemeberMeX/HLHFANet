# HLHFANet
This is a re-implementation of our paper and for non-commercial use only. 
----------------------------------------------------------------
dataset    Rain100H   Rain 100L   Rain12    Rain1400    Rain800
----------------------------------------------------------------
SSIM       0.906       0.983	    0.964      0.950       0.902
----------------------------------------------------------------
PSNR       30.24       38.50	    36.85      32.61       27.41
----------------------------------------------------------------
FSIM       0.923       0.984      0.854      0.964       0.933
----------------------------------------------------------------
You need to install Python with Pytorch-GPU to run this code.

Usage:

1. Preparing training data: put rainy images into "/input"

2.Run
1）You can run testrain100H.py and testreal.py, model is saved in "/logs/Rain100H"; You can run testrain100L.py and testrain12.py, model is saved in "/logs/Rain100L";
    You can run testrain1400.py, model is saved in "/logs/Rain1400"; You can run testrain800.py, model is saved in "/logs/Rain800".

2）Rain100H and Realrain use the same training model Rain100H.

3）Rain100L and Rain12 use the same training model Rain100L.

