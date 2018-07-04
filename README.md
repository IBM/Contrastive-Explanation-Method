# Contrastive-Explanation-Method
Codes for reproducing the contrastive explanation in  “[Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives](https://arxiv.org/abs/1802.07623)”


To find the pertinent positive (PP) of an instance, 

```shell
python3 main.py -i 2953 --mode PP --kappa 10 --gamma 100
```
This would find the PP of image ID 2953 in the test images from the MNIST dataset.

<img src="/Results/PP_ID2953_Gamma_100.0/Orig_original5.png" width="80" height="80"> <img src="/Results/PP_ID2953_Gamma_100.0/Delta_id2953_kappa10.0_Orig5_Adv3_Delta5.png" width="80" height="80">


From left to right: the original image and the pertinent positive component. This PP in Image 2953 is sufficient to be classified as 5.

To find the pertinent negative (PN) of an instance,

```shell
python3 main.py -i 340 --mode PN --kappa 10 --gamma 100
```
This would find the PN of image ID 340 in the test images from the MNIST dataset.

<img src="/Results/PN_ID340_Gamma_100.0/Orig_original3.png" width="80" height="80"><img src="/Results/PN_ID340_Gamma_100.0/Delta_id340_kappa10.0_Orig3_Adv5_Delta8.png" width="80" height="80"><img src="/Results/PN_ID340_Gamma_100.0/Adv_id340_kappa10.0_Orig3_Adv5_Delta8.png" width="80" height="80">


From left to right: the original image, the pertinent negative component and the image composed of the original image and PN. If we add PN to Image 340, it would be classified as 5.

The argument `kappa` (confidence lebel) and `gamma` (regularization coefficient of autoencoder) are tuning parameters for the optimization setup. Both PP and PN are used to explain the model prediction results. For more details, please refer to the paper.
