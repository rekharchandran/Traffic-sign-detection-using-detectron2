# Trafiic Sign-detection-using-Detectron2

![14](https://user-images.githubusercontent.com/50706192/151142886-f1c4bf43-00f2-4ab4-93ae-741467c8a38f.png)



An object detection model using traffic signs custom dataset from scratch
 
 Detectron 2 comes to the rescue if we want to train an object detection model in a snap with a custom dataset. All the models present in the model zoo of the Detectron 2 library are pre-trained on COCO Dataset. We just need to fine-tune our custom dataset on the pre-trained model.
 
 Detectron 2 is a complete rewrite of the first Detectron which was released in the year 2018. The predecessor was written on Caffe2, a deep learning framework that is also backed by Facebook. Both the Caffe2 and Detectron are now deprecated. Caffe2 is now a part of PyTorch and the successor, Detectron 2 is completely written on PyTorch.
 
 Let’s dive into Instance Detection directly.
Instance Detection refers to the classification and localization of an object with a bounding box around it.  We are going to deal with identifying the traffic signs from images using the Faster RCNN model and instance segmentatin(Mask-RCNN)from the Detectron 2’s model zoo.
Note that we are going to limit our class by 15.

  Step 1: Installing Detectron 2
    If you have any doubts how toinstall detectron2 in windows please follow this tutorial TheCodingBug https://www.youtube.com/watch?v=Pb3opEFP94U&t=151s 
    
   Step2 : Prepare and Register the Dataset
   
   Step3 : Visualize the Training set
   
   Step4 : Training the Model
   
   Step5 : Inference using the Trained Model
    
   

