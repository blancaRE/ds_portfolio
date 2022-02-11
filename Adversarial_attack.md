## Load and use the model


```python
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution() 
import keras
import matplotlib.pyplot as plt

```


```python
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras import backend as K
from keras.preprocessing import image #To load images
```


```python
# Load InceptionV3 model
model = InceptionV3()
```

    WARNING:tensorflow:From C:\Users\blara\miniconda3\lib\site-packages\keras\layers\normalization\batch_normalization.py:532: _colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    

To use inception v3 model, the images must meet certain <a href="https://keras.rstudio.com/reference/application_inception_v3.html">requirements</a>:
* input_shape: (299, 299, 3)
* Pixel intensity range: [-1,1] 
* Inception allows multiple inputs. The first input should de the number of images.


```python
#Load and re-scale image
x = image.img_to_array(image.load_img("./table_00.jpg",target_size=(299,299)))
# Change pixel intensity from RGB scale ([0,255]) to [-1, 1]
x /= 255
x -= 0.5
x *= 2
number_images=1
my_image=x.reshape([number_images, x.shape[0], x.shape[1], x.shape[2]])
#Chek input size
print(my_image.shape)

```

    (1, 299, 299, 3)
    

The output of the model is a vector of size 1000, every entry corresponds to the probability of belonging to one of the 1000 classes.


```python
y = model.predict(my_image)
y.shape
```

    C:\Users\blara\miniconda3\lib\site-packages\keras\engine\training_v1.py:2079: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
      updates=self.state_updates,
    




    (1, 1000)




```python
result=decode_predictions(y)# Retunr (class_name, class_description, score)
result=result[0]
plt.figure(figsize=(15, 3))
plt.title('Probability of the first 5 classes')    
for a, b, c in result:
    plt.bar(b, c)
```


    
![png](output_8_0.png)
    


## Adversarial attack
Select a target class and a minimum likelihood for the target class. Define cost function maximize the likelihood of the target class (given by the last output layer). Define gradient: : we do not want to modify the parameters of the model as uasual, but the input variable. 


```python
inp_layer = model.layers[0].input 
out_layer = model.layers[-1].output 
target_class = 929 # ice-cream
likelihood_target_class= 0.90
# Cost function: 
loss = out_layer[0, target_class]
# Gradient:
grad= K.gradients(loss, inp_layer)[0] 
optimize_gradient = K.function([inp_layer, K.learning_phase()], [grad, loss]) # [input variable] [outputvariable]

```

### First try

Iterative loop to modify the input image. The optimize_gradient returns a tensor whose values set how to modify the pixels of the input image to optimize the cost function, and therefore maximize the likelihood of the target class.


```python
my_image_hacked1 = np.copy(my_image) # new hacked image
cost= 0.0
while cost< likelihood_target_class: 
    gr, cost = optimize_gradient([my_image_hacked1,0]) # guardamos el gradiente en cda una de las operaciones. entrada: imagen x (copia)
    # una vez hehco esto, en el gradiente estan los valores de los pixeles. Se los sumamos a nuestra imagen
    my_image_hacked1 += gr
    print("Target cost:", cost)
```

    Target cost: 0.00012288845
    Target cost: 0.0001240762
    Target cost: 0.00012529078
    Target cost: 0.0001265163
    Target cost: 0.00012773502
    Target cost: 0.00012894027
    Target cost: 0.0001301377
    Target cost: 0.00013132149
    Target cost: 0.00013250673
    Target cost: 0.00013368571
    Target cost: 0.00013486753
    Target cost: 0.00013607278
    Target cost: 0.00013727065
    Target cost: 0.0001384758
    Target cost: 0.00013968286
    Target cost: 0.00014087143
    Target cost: 0.00014206312
    Target cost: 0.00014326427
    Target cost: 0.00014446207
    Target cost: 0.00014566525
    Target cost: 0.0001468644
    Target cost: 0.00014805788
    Target cost: 0.00014924198
    Target cost: 0.00015043415
    Target cost: 0.0001516316
    Target cost: 0.00015284769
    Target cost: 0.0001540854
    Target cost: 0.00015531878
    Target cost: 0.00015654531
    Target cost: 0.00015778172
    Target cost: 0.00015903074
    Target cost: 0.0001602747
    Target cost: 0.00016153253
    Target cost: 0.00016280271
    Target cost: 0.00016408834
    Target cost: 0.00016539145
    Target cost: 0.00016670268
    Target cost: 0.00016802929
    Target cost: 0.00016936747
    Target cost: 0.00017071381
    Target cost: 0.00017207739
    Target cost: 0.000173476
    Target cost: 0.00017487377
    Target cost: 0.0001762763
    Target cost: 0.0001776726
    Target cost: 0.00017907404
    Target cost: 0.00018052816
    Target cost: 0.00018200386
    Target cost: 0.00018349377
    Target cost: 0.0001849998
    Target cost: 0.00018653151
    Target cost: 0.0001880869
    Target cost: 0.00018966681
    Target cost: 0.00019122947
    Target cost: 0.00019278709
    Target cost: 0.00019436494
    Target cost: 0.00019596696
    Target cost: 0.00019758097
    Target cost: 0.0001991955
    Target cost: 0.0002008532
    Target cost: 0.00020252266
    Target cost: 0.00020423958
    Target cost: 0.00020595879
    Target cost: 0.00020770586
    Target cost: 0.00020947043
    Target cost: 0.00021126476
    Target cost: 0.00021306163
    Target cost: 0.00021486601
    Target cost: 0.00021670686
    Target cost: 0.00021858752
    Target cost: 0.0002204762
    Target cost: 0.00022237828
    Target cost: 0.00022429325
    Target cost: 0.00022620361
    Target cost: 0.00022812831
    Target cost: 0.00023005895
    Target cost: 0.0002320222
    Target cost: 0.00023400158
    Target cost: 0.00023596946
    Target cost: 0.00023796191
    Target cost: 0.00023996033
    Target cost: 0.00024199761
    Target cost: 0.00024410461
    Target cost: 0.00024622877
    Target cost: 0.0002483549
    Target cost: 0.00025047964
    Target cost: 0.00025262122
    Target cost: 0.0002547895
    Target cost: 0.00025697637
    Target cost: 0.00025922837
    Target cost: 0.00026151902
    Target cost: 0.00026380434
    Target cost: 0.00026610828
    Target cost: 0.0002684548
    Target cost: 0.000270868
    Target cost: 0.00027330173
    Target cost: 0.00027574043
    Target cost: 0.00027819682
    Target cost: 0.0002806884
    Target cost: 0.00028322337
    Target cost: 0.00028577587
    Target cost: 0.00028834757
    Target cost: 0.00029089296
    Target cost: 0.00029340218
    Target cost: 0.00029587865
    Target cost: 0.00029836193
    Target cost: 0.00030086967
    Target cost: 0.00030340726
    Target cost: 0.0003059497
    Target cost: 0.00030851297
    Target cost: 0.00031111934
    Target cost: 0.00031370277
    Target cost: 0.00031625348
    Target cost: 0.0003188094
    Target cost: 0.00032137646
    Target cost: 0.00032401696
    Target cost: 0.00032668412
    Target cost: 0.00032936118
    Target cost: 0.00033205264
    Target cost: 0.00033472918
    Target cost: 0.0003374248
    Target cost: 0.00034015544
    Target cost: 0.00034288748
    Target cost: 0.00034561774
    Target cost: 0.0003483637
    Target cost: 0.0003511611
    Target cost: 0.0003539627
    Target cost: 0.00035679428
    Target cost: 0.0003596702
    Target cost: 0.00036255945
    Target cost: 0.00036548247
    Target cost: 0.0003684644
    Target cost: 0.00037145434
    Target cost: 0.00037450626
    Target cost: 0.00037755587
    Target cost: 0.00038062863
    Target cost: 0.000383663
    Target cost: 0.0003866479
    Target cost: 0.00038963047
    Target cost: 0.00039268486
    Target cost: 0.00039578875
    Target cost: 0.00039892594
    Target cost: 0.00040213828
    Target cost: 0.000405371
    Target cost: 0.0004086619
    Target cost: 0.00041199478
    Target cost: 0.00041533576
    Target cost: 0.0004187648
    Target cost: 0.00042219862
    Target cost: 0.00042563645
    Target cost: 0.0004291102
    Target cost: 0.00043261325
    Target cost: 0.00043610958
    Target cost: 0.00043961214
    Target cost: 0.00044313737
    Target cost: 0.00044668507
    Target cost: 0.0004502971
    Target cost: 0.00045397712
    Target cost: 0.0004576169
    Target cost: 0.0004612961
    Target cost: 0.00046500555
    Target cost: 0.00046873806
    Target cost: 0.000472504
    Target cost: 0.00047633002
    Target cost: 0.00048023553
    Target cost: 0.00048411908
    Target cost: 0.00048804167
    Target cost: 0.00049208174
    Target cost: 0.0004961634
    Target cost: 0.0005003011
    Target cost: 0.00050453085
    Target cost: 0.00050884386
    Target cost: 0.00051322574
    Target cost: 0.0005176664
    Target cost: 0.00052214024
    Target cost: 0.0005267183
    Target cost: 0.00053174846
    Target cost: 0.0005367779
    Target cost: 0.00054190523
    Target cost: 0.0005469954
    Target cost: 0.00055198494
    Target cost: 0.00055709784
    Target cost: 0.000562326
    Target cost: 0.0005675639
    Target cost: 0.0005728837
    Target cost: 0.0005782712
    Target cost: 0.0005836134
    Target cost: 0.0005889761
    Target cost: 0.0005944407
    Target cost: 0.00060003076
    Target cost: 0.00060566596
    Target cost: 0.00061140826
    Target cost: 0.00061724865
    Target cost: 0.00062304386
    Target cost: 0.00062887924
    Target cost: 0.00063492457
    Target cost: 0.00064098433
    Target cost: 0.000647121
    Target cost: 0.0006533512
    Target cost: 0.0006598032
    Target cost: 0.0006664143
    Target cost: 0.00067305245
    Target cost: 0.0006799578
    Target cost: 0.00068726455
    Target cost: 0.0006948702
    Target cost: 0.0007028014
    Target cost: 0.000710911
    Target cost: 0.00071921595
    Target cost: 0.00072783476
    Target cost: 0.0007370839
    Target cost: 0.00074700103
    Target cost: 0.00075720175
    Target cost: 0.0007675668
    Target cost: 0.0007781742
    Target cost: 0.000789487
    Target cost: 0.00080186356
    Target cost: 0.00081589376
    Target cost: 0.00083214795
    Target cost: 0.0008561188
    Target cost: 0.00088482845
    Target cost: 0.0009173197
    Target cost: 0.00095493504
    Target cost: 0.0010003976
    Target cost: 0.0010498357
    Target cost: 0.0010997811
    Target cost: 0.0011595306
    Target cost: 0.001226424
    Target cost: 0.0012950742
    Target cost: 0.0013798899
    Target cost: 0.0014527374
    Target cost: 0.0015446678
    Target cost: 0.0015256278
    Target cost: 0.001822121
    Target cost: 0.001996777
    Target cost: 0.0021626442
    Target cost: 0.0021793956
    Target cost: 0.0025099982
    Target cost: 0.0030292277
    Target cost: 0.003016495
    Target cost: 0.0037641535
    Target cost: 0.0037790253
    Target cost: 0.0046877996
    Target cost: 0.005873
    Target cost: 0.0047690077
    Target cost: 0.010790629
    Target cost: 0.0022914025
    Target cost: 0.0074988324
    Target cost: 0.009556317
    Target cost: 0.012056084
    Target cost: 0.006247825
    Target cost: 0.02285105
    Target cost: 0.005499892
    Target cost: 0.15230927
    Target cost: 0.0022688825
    Target cost: 0.0062264786
    Target cost: 0.044413764
    Target cost: 0.051113207
    Target cost: 0.054538846
    Target cost: 0.29524136
    Target cost: 0.0096970815
    Target cost: 0.102924764
    Target cost: 0.90313447
    


```python
grad= K.gradients(loss, inp_layer)[0] 
```


```python
y = model.predict(my_image_hacked1)
result=decode_predictions(y)# Retunr (class_name, class_description, score)
result=result[0]
plt.figure(figsize=(15, 3))
plt.title('Probability of the first 5 classes')    
for a, b, c in result:
    plt.bar(b, c)
```


    
![png](output_15_0.png)
    


Visualize hacked photo


```python
my_image_hacked1 /= 2
my_image_hacked1 += 0.5
my_image_hacked1 *= 255
plt.imshow(my_image_hacked1[0].astype(np.uint8)) 
```




    <matplotlib.image.AxesImage at 0x13f8e546d90>




    
![png](output_17_1.png)
    


### Second try
The class with maximum likelihood is the target class, but the changes are visible to human aye. Add a maximum perturbation to makes changes invisible to human eye.


```python
pert= 0.01
max_pert = x + pert
min_pert = x - pert
```


```python
my_image_hacked2 = np.copy(my_image) # new hacked image
cost= 0.0
while cost< likelihood_target_class: 
    gr, cost = optimize_gradient([my_image_hacked2,0]) 
    my_image_hacked2 += gr
    # New code to set maximum perturbation
    my_image_hacked2 = np.clip(my_image_hacked2, min_pert, max_pert) 
    my_image_hacked2 = np.clip(my_image_hacked2,-1, 1) 
    print("Target cost:", cost)
```

    Target cost: 0.00012288845
    Target cost: 0.00012407632
    Target cost: 0.00012529075
    Target cost: 0.00012651614
    Target cost: 0.00012773488
    Target cost: 0.0001289402
    Target cost: 0.0001301375
    Target cost: 0.00013132133
    Target cost: 0.00013250657
    Target cost: 0.00013368553
    Target cost: 0.00013486735
    Target cost: 0.0001360725
    Target cost: 0.00013726979
    Target cost: 0.00013847464
    Target cost: 0.00013968212
    Target cost: 0.0001408698
    Target cost: 0.00014206172
    Target cost: 0.00014326208
    Target cost: 0.00014446062
    Target cost: 0.00014566425
    Target cost: 0.00014686429
    Target cost: 0.00014805821
    Target cost: 0.00014924209
    Target cost: 0.00015043306
    Target cost: 0.0001516307
    Target cost: 0.00015284645
    Target cost: 0.00015408259
    Target cost: 0.00015531624
    Target cost: 0.00015654207
    Target cost: 0.00015777948
    Target cost: 0.00015902884
    Target cost: 0.0001602724
    Target cost: 0.000161529
    Target cost: 0.0001627967
    Target cost: 0.00016408204
    Target cost: 0.00016538524
    Target cost: 0.00016669667
    Target cost: 0.00016802577
    Target cost: 0.0001693628
    Target cost: 0.00017070836
    Target cost: 0.00017207277
    Target cost: 0.00017346989
    Target cost: 0.00017486869
    Target cost: 0.00017627206
    Target cost: 0.00017766778
    Target cost: 0.00017907404
    Target cost: 0.00018052573
    Target cost: 0.00018200005
    Target cost: 0.00018349028
    Target cost: 0.00018499269
    Target cost: 0.00018652475
    Target cost: 0.000188081
    Target cost: 0.00018966243
    Target cost: 0.00019122406
    Target cost: 0.00019278211
    Target cost: 0.00019435935
    Target cost: 0.000195962
    Target cost: 0.00019757448
    Target cost: 0.00019918985
    Target cost: 0.0002008404
    Target cost: 0.0002025106
    Target cost: 0.00020421155
    Target cost: 0.0002059333
    Target cost: 0.00020768441
    Target cost: 0.00020944736
    Target cost: 0.00021124462
    Target cost: 0.00021303468
    Target cost: 0.00021484046
    Target cost: 0.00021668668
    Target cost: 0.00021856652
    Target cost: 0.00022045507
    Target cost: 0.00022236911
    Target cost: 0.00022428836
    Target cost: 0.0002262026
    Target cost: 0.00022811374
    Target cost: 0.00023004286
    Target cost: 0.0002320126
    Target cost: 0.00023399333
    Target cost: 0.00023596648
    Target cost: 0.00023795143
    Target cost: 0.00023995055
    Target cost: 0.00024198808
    Target cost: 0.00024408539
    Target cost: 0.00024620479
    Target cost: 0.0002483277
    Target cost: 0.00025045275
    Target cost: 0.0002525973
    Target cost: 0.00025476218
    Target cost: 0.0002569472
    Target cost: 0.00025920916
    Target cost: 0.0002614995
    Target cost: 0.0002637855
    Target cost: 0.00026610034
    Target cost: 0.00026843476
    Target cost: 0.0002708454
    Target cost: 0.00027327737
    Target cost: 0.00027571572
    Target cost: 0.00027817278
    Target cost: 0.00028065598
    Target cost: 0.00028319503
    Target cost: 0.00028575526
    Target cost: 0.0002883247
    Target cost: 0.0002908724
    Target cost: 0.00029338617
    Target cost: 0.00029586285
    Target cost: 0.00029835245
    Target cost: 0.0003008607
    Target cost: 0.00030339655
    Target cost: 0.00030595565
    Target cost: 0.00030852185
    Target cost: 0.00031112763
    Target cost: 0.00031371208
    Target cost: 0.0003162402
    Target cost: 0.000318796
    Target cost: 0.00032137294
    Target cost: 0.00032400186
    Target cost: 0.0003266696
    Target cost: 0.00032935702
    Target cost: 0.00033205518
    Target cost: 0.0003347404
    Target cost: 0.00033744497
    Target cost: 0.0003401802
    Target cost: 0.00034291967
    Target cost: 0.00034565036
    Target cost: 0.00034840664
    Target cost: 0.00035118984
    Target cost: 0.00035398421
    Target cost: 0.0003568042
    Target cost: 0.00035963836
    Target cost: 0.00036250995
    Target cost: 0.0003654312
    Target cost: 0.00036840452
    Target cost: 0.0003713894
    Target cost: 0.00037443684
    Target cost: 0.0003774892
    Target cost: 0.00038055776
    Target cost: 0.00038360522
    Target cost: 0.00038657253
    Target cost: 0.00038955276
    Target cost: 0.0003926078
    Target cost: 0.00039569798
    Target cost: 0.00039881983
    Target cost: 0.00040202335
    Target cost: 0.00040525963
    Target cost: 0.00040852217
    Target cost: 0.00041184083
    Target cost: 0.00041516742
    Target cost: 0.00041858354
    Target cost: 0.00042202152
    Target cost: 0.00042546247
    Target cost: 0.0004289403
    Target cost: 0.0004324114
    Target cost: 0.00043589622
    Target cost: 0.0004394092
    Target cost: 0.000442894
    Target cost: 0.00044644641
    Target cost: 0.00045005372
    Target cost: 0.0004537132
    Target cost: 0.0004573914
    Target cost: 0.0004610704
    Target cost: 0.00046479213
    Target cost: 0.0004685367
    Target cost: 0.00047228608
    Target cost: 0.00047610834
    Target cost: 0.0004800176
    Target cost: 0.00048392877
    Target cost: 0.00048783718
    Target cost: 0.0004918202
    Target cost: 0.0004958846
    Target cost: 0.0005000385
    Target cost: 0.0005042541
    Target cost: 0.00050855457
    Target cost: 0.00051293767
    Target cost: 0.00051735644
    Target cost: 0.0005218717
    Target cost: 0.0005264107
    Target cost: 0.0005310605
    Target cost: 0.0005361685
    Target cost: 0.00054130243
    Target cost: 0.0005464255
    Target cost: 0.00055141596
    Target cost: 0.00055652455
    Target cost: 0.000561736
    Target cost: 0.0005669526
    Target cost: 0.0005722568
    Target cost: 0.00057763146
    Target cost: 0.0005830793
    Target cost: 0.00058845035
    Target cost: 0.0005939634
    Target cost: 0.0005994951
    Target cost: 0.0006052458
    Target cost: 0.00061093946
    Target cost: 0.00061669643
    Target cost: 0.00062248524
    Target cost: 0.0006283097
    Target cost: 0.0006342316
    Target cost: 0.0006403153
    Target cost: 0.00064648036
    Target cost: 0.0006528031
    Target cost: 0.0006591609
    Target cost: 0.00066576584
    Target cost: 0.0006722597
    Target cost: 0.00067902316
    Target cost: 0.00068603683
    Target cost: 0.00069330493
    Target cost: 0.00070103374
    Target cost: 0.00070906454
    Target cost: 0.000717364
    Target cost: 0.00072594173
    Target cost: 0.00073487073
    Target cost: 0.0007444369
    Target cost: 0.00075476075
    Target cost: 0.00076493
    Target cost: 0.0007754396
    Target cost: 0.0007863014
    Target cost: 0.00079812296
    Target cost: 0.00081128225
    Target cost: 0.0008258137
    Target cost: 0.0008431024
    Target cost: 0.0008687695
    Target cost: 0.00089708756
    Target cost: 0.0009290213
    Target cost: 0.0009666148
    Target cost: 0.0010135073
    Target cost: 0.0010654705
    Target cost: 0.0011190303
    Target cost: 0.0011815603
    Target cost: 0.0012462803
    Target cost: 0.0013137832
    Target cost: 0.0013847782
    Target cost: 0.0014556271
    Target cost: 0.001580543
    Target cost: 0.0017124865
    Target cost: 0.001846367
    Target cost: 0.0019591793
    Target cost: 0.001980111
    Target cost: 0.002284008
    Target cost: 0.002564147
    Target cost: 0.0030514554
    Target cost: 0.003178741
    Target cost: 0.0028132426
    Target cost: 0.0049517434
    Target cost: 0.0034506982
    Target cost: 0.006706546
    Target cost: 0.004830841
    Target cost: 0.009489739
    Target cost: 0.0050907517
    Target cost: 0.016278286
    Target cost: 0.003483534
    Target cost: 0.016278362
    Target cost: 0.004554367
    Target cost: 0.043216895
    Target cost: 0.0081050955
    Target cost: 0.09032355
    Target cost: 0.0039037596
    Target cost: 0.030702207
    Target cost: 0.14649732
    Target cost: 0.014076095
    Target cost: 0.6353232
    Target cost: 0.005699895
    Target cost: 0.056518
    Target cost: 0.1439174
    Target cost: 0.15236792
    Target cost: 0.41333798
    Target cost: 0.065937854
    Target cost: 0.8577718
    Target cost: 0.2529702
    Target cost: 0.32303372
    Target cost: 0.28742066
    Target cost: 0.32529902
    Target cost: 0.16932352
    Target cost: 0.80760396
    Target cost: 0.6713045
    Target cost: 0.17127806
    Target cost: 0.81417334
    Target cost: 0.35503295
    Target cost: 0.39521763
    Target cost: 0.21044502
    Target cost: 0.74653614
    Target cost: 0.5508852
    Target cost: 0.38869184
    Target cost: 0.27714652
    Target cost: 0.79309803
    Target cost: 0.6256245
    Target cost: 0.37621146
    Target cost: 0.42497697
    Target cost: 0.54338413
    Target cost: 0.40046307
    Target cost: 0.5238165
    Target cost: 0.3392143
    Target cost: 0.78758043
    Target cost: 0.7917767
    Target cost: 0.7204079
    Target cost: 0.3387654
    Target cost: 0.42174214
    Target cost: 0.13686001
    Target cost: 0.9119743
    


```python
grad= K.gradients(loss, inp_layer)[0] 
```


```python
y = model.predict(my_image_hacked2)
result=decode_predictions(y)# Retunr (class_name, class_description, score)
result=result[0]
plt.figure(figsize=(15, 3))
plt.title('Probability of the first 5 classes')    
for a, b, c in result:
    plt.bar(b, c)
```


    
![png](output_22_0.png)
    


Visualize hacked photo


```python
my_image_hacked2 /= 2
my_image_hacked2 += 0.5
my_image_hacked2 *= 255
plt.imshow(my_image_hacked2[0].astype(np.uint8)) # image show
```




    <matplotlib.image.AxesImage at 0x13f90cc6460>




    
![png](output_24_1.png)
    



```python

```
