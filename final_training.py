#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
print(torch.__version__)


# In[2]:


get_ipython().run_line_magic('cd', 'detectron2')


# In[3]:


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random,pickle

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog,build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import datasets
from detectron2 import model_zoo


# In[4]:


torch.cuda.is_available()


# In[5]:


def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(), plt.imshow(im), plt.axis('off');


# In[6]:


import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from matplotlib.pyplot import imshow
from PIL import Image
import IPython
def cv2_imshow(img):
    img = img[:,:,[2,1,0]]
    img = Image.fromarray(img)
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# In[7]:


dataset_name =  "/storage/Fylkesveg/Sign_training_material/training_sahi"


# In[8]:


train_dataset_name = "trafficsigns_train"
train_image_path =  "/storage/Fylkesveg/Sign_training_material/training_sahi/train/coco_images_800_02"
train_json_annot_path = "/storage/Fylkesveg/Sign_training_material/training_sahi/train/coco-with-negatives.json"
f = open(train_json_annot_path, 'r')
train_annotations = json.load(f)
f.close()
num_classes = len(train_annotations['categories'])


# In[9]:


cfg_save_path =  'IS_cfg.pickle'
training_dict = datasets.load_coco_json(train_json_annot_path, train_image_path,
                dataset_name=train_dataset_name)


# In[10]:


test_dataset_name = 'trafficsigns_valid'
test_image_path = "/storage/Fylkesveg/Sign_training_material/training_sahi/valid/coco_images_800_02"
test_json_annot_path = "/storage/Fylkesveg/Sign_training_material/training_sahi/valid/coco-with-negatives.json"
f = open(test_json_annot_path, 'r')
test_annotations = json.load(f)
f.close()
num_classes = len(test_annotations['categories'])


# In[11]:


from detectron2.data.datasets import register_coco_instances
register_coco_instances(train_dataset_name, {}, train_json_annot_path, train_image_path)
register_coco_instances(test_dataset_name, {}, test_json_annot_path, test_image_path)


# In[12]:


trafficsigns_metadata = MetadataCatalog.get("trafficsigns_train")
dataset_dicts = DatasetCatalog.get("trafficsigns_train")


# In[13]:


print(type(trafficsigns_metadata))
MetadataCatalog.get("trafficsigns_train")


# In[14]:


cfg = get_cfg()
device = 'cuda'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("trafficsigns_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
#cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl"  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS =  model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR =  0.00025 
cfg.SOLVER.MAX_ITER = 270000  
cfg.SOLVER.STEPS = (210000, 250000)       
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.INPUT.MIN_SIZE_TRAIN = [640,672,704,736,768,800]
cfg.OUTPUT_DIR = 'output/trafficsigns_train_X101-FPN/'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# In[15]:


yaml_cfg_path = 'output/trafficsigns_X101-FPN/cfg.yaml'
with open(yaml_cfg_path, 'w') as yaml_cfg_file:
  yaml_cfg_file.write(cfg.dump())


# In[50]:


outputs


# In[16]:


trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


# In[23]:


get_ipython().system('kill 27747')


# In[24]:


# Look at training curves in tensorboard:
get_ipython().run_line_magic('reload_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir output')


# In[25]:


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("trafficsigns_train", )
predictor = DefaultPredictor(cfg)


# In[27]:


from detectron2.utils.visualizer import ColorMode
dataset_dicts = DatasetCatalog.get("trafficsigns_train")
for d in random.sample(dataset_dicts, 10):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=trafficsigns_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])


# In[28]:


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("trafficsigns_train", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "trafficsigns_train")
print(inference_on_dataset(predictor.model, val_loader, evaluator))


# In[29]:


#Validation Sample


# In[ ]:


test_dataset_name = 'trafficsigns_valid'
test_image_path = "/storage/Fylkesveg/Sign_training_material/training_sahi/valid/coco_images_800_02"
test_json_annot_path = "/storage/Fylkesveg/Sign_training_material/training_sahi/valid/coco-with-negatives.json"
f = open(test_json_annot_path, 'r')
test_annotations = json.load(f)
f.close()
num_classes = len(test_annotations['categories'])


# In[30]:


trafficsigns_metadata = MetadataCatalog.get("trafficsigns_valid")
dataset_dicts = DatasetCatalog.get("trafficsigns_valid")


# In[31]:


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("trafficsigns_valid", )
predictor = DefaultPredictor(cfg)


# In[32]:


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("trafficsigns_valid", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "trafficsigns_valid")
print(inference_on_dataset(predictor.model, val_loader, evaluator))


# In[34]:


from detectron2.utils.visualizer import ColorMode
dataset_dicts = DatasetCatalog.get("trafficsigns_valid")
for d in random.sample(dataset_dicts, 10):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=trafficsigns_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])


# ###  Confusion Matrix from scratch

# In[54]:


get_ipython().run_line_magic('cd', '"/srv/notebooks/detectron2/object_detection_confusion_matrix"')


# In[91]:


from confusion_matrix import ConfusionMatrix
cm = ConfusionMatrix(20, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5)
for d in dataset_dicts_validation:
    if len(d["annotations"]) < 1:
        continue
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    labels = list()
    detections = list()
    for ann in d["annotations"]:
        bbox = [
            ann["bbox"][0], # X1
            ann["bbox"][1], # Y1
            ann["bbox"][0] + ann["bbox"][2], #X2
            ann["bbox"][1] + ann["bbox"][3], #Y2
        ]
        labels.append([ann["category_id"]] + bbox)
    for coord, conf, cls in zip(
        outputs["instances"].get("pred_boxes").tensor.cpu().numpy(), 
        outputs["instances"].get("scores").cpu().numpy(), 
        outputs["instances"].get("pred_classes").cpu().numpy()
    ):
        detections.append(list(coord) + [conf] + [cls])
    cm.process_batch(np.array(detections), np.array(labels))


# In[109]:


categories = MetadataCatalog.get("trafficsigns_train").thing_classes
row_labels = categories + ['extra row']
col_labels = categories + ['extra col']
row_labels


# In[100]:


d = dataset_dicts_validation[0]
img = cv2.imread(d["file_name"])
outputs = predictor(img)
labels = list()
detections = list()
for ann in d["annotations"]:
    bbox = [
        ann["bbox"][0], # X1
        ann["bbox"][1], # Y1
        ann["bbox"][0] + ann["bbox"][2], #X2
        ann["bbox"][1] + ann["bbox"][3], #Y2
    ]
    labels.append([ann["category_id"]] + bbox)
for coord, conf, cls in zip(
    outputs["instances"].get("pred_boxes").tensor.cpu().numpy(), 
    outputs["instances"].get("scores").cpu().numpy(), 
    outputs["instances"].get("pred_classes").cpu().numpy()
):
    detections.append(list(coord) + [conf] + [cls])
cm.process_batch(np.array(detections), np.array(labels))


# In[110]:


import pandas as pd
df = pd.DataFrame(cm.matrix, columns=col_labels, index=row_labels)
df


# In[ ]:




