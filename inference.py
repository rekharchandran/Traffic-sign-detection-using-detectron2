from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import datasets
import os
import pickle
import cv2
from tqdm import tqdm
from detectron2.utils import *
#from utils import *
from glob import glob


from sahi.model import Detectron2DetectionModel
from sahi.predict import get_sliced_prediction
# from sahi.utils.file import download_from_url
from sahi.utils.cv import visualize_object_predictions
from IPython.display import Image

model_path = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
base_image_path = '/storage/Projects/Fylkesveg/Sign_training_material/FV7768_Ladybug_Grøtfjord_1'
train_dataset_name = 'trafficsigns'
train_image_path = '/storage/Projects/Fylkesveg/Sign_training_material/training_sahi/images/train/'
train_json_annot_path = '/storage/Projects/Fylkesveg/Sign_training_material/training_sahi/labels/train/coco.json'
training_dict = datasets.load_coco_json(train_json_annot_path, train_image_path,
                dataset_name=train_dataset_name)
register_coco_instances(train_dataset_name, {}, train_json_annot_path, train_image_path)

yaml_cfg_path = os.path.join('/storage/Projects/notebooks/detectron2/detectron2/output/trafficsigns_X101-FPN','cfg.yaml')

detection_model = Detectron2DetectionModel(
  model_path = os.path.join( '/storage/Projects/notebooks/detectron2/detectron2/output/trafficsigns_X101-FPN', 'model_final.pth'),
  config_path = yaml_cfg_path,
  image_size = 640,
  device = 'cuda:0'
)

findings = open(os.path.join(base_image_path,'findings.csv'), 'w')
findings.write('image,category,bbox_x1,bbox_y1,bbox_x2,bbox_y2,score\n')
findings.flush()
# images = glob('/home/stian/Data/Fylkesveg/Sign_training_material/FV7768_Ladybug_Grøtfjord_1/ladybug*.jpg')
images = [
  os.path.join(base_image_path,'ladybugImageOutput_00000009.jpg'),
  os.path.join(base_image_path,'ladybugImageOutput_00000010.jpg'),
  os.path.join(base_image_path,'ladybugImageOutput_00000070.jpg'),
  os.path.join(base_image_path,'ladybugImageOutput_00000071.jpg'),
  os.path.join(base_image_path,'ladybugImageOutput_00000072.jpg'),
  os.path.join(base_image_path,'ladybugImageOutput_00000195.jpg'),
  os.path.join(base_image_path,'ladybugImageOutput_00000271.jpg'),
  os.path.join(base_image_path,'ladybugImageOutput_00000272.jpg'),
  os.path.join(base_image_path,'ladybugImageOutput_00000297.jpg')
]
for image_path in tqdm(images):
  image = cv2.imread(image_path)
  result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 800,
    slice_width = 800,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
    verbose = 0
  )
  if(len(result.object_prediction_list) > 0):
    labeled_image = visualize_object_predictions(
      image = image,
      object_prediction_list = result.object_prediction_list
    )
    for p in result.object_prediction_list:
      if(p.score.value < 0.9):
        continue
      findings.write(f'{os.path.basename(image_path)},{p.category.name},{p.bbox.minx},{p.bbox.miny},{p.bbox.maxx},{p.bbox.maxy},{p.score.value}\n')
      findings.flush()
    #  cv2.imwrite(os.path.join(os.path.dirname(image_path),'labeled-'+os.path.basename(image_path)), labeled_image['image'])
    #cv2.imshow(image_path, labeled_image['image'])
    #cv2.waitKey()
    cv2.imwrite(os.path.join(os.path.dirname(image_path),'labeled-'+os.path.basename(image_path)), labeled_image['image'])
findings.close()
