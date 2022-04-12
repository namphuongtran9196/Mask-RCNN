import os
import json
import argparse
import tqdm
# Teminal arguments
parser = argparse.ArgumentParser(description='Convert dataset from coco dataset')
parser.add_argument('-i','--input',type=str,help='path to coco dataset json file',
                    default='../../data/coco/annotations_trainval2014/annotations/instances_train2014.json')
parser.add_argument('-o','--output',type=str,help='path to output directory',
                    default='../../data/coco_converted')

def main(args):
    with open(args.input) as f:
        database = json.load(f)
    dataset = {}
    os.makedirs(args.output,exist_ok=True)
    
    print('Writing categories')
    # Get categories
    value = database['categories']
    # Serializing json 
    json_object = json.dumps(value, indent = 4)
    # Writing to sample.json
    with open(os.path.join(args.output,'categories.json'), "w") as outfile:
        outfile.write(json_object)
        
        
    args.output = os.path.join(args.output,'annotations')
    os.makedirs(args.output,exist_ok=True)
    print('Creating database...')
    for image_info in tqdm.tqdm(database['images']):
        image_id = image_info['id']
        dataset[image_id] = {'image_id' : image_info['id'],
                            'file_name' : image_info['file_name'],
                             'width' : image_info['width'],
                             'height': image_info['height'],
                             'annotations' : []}
    print('Converting database....')
    for annotation in tqdm.tqdm(database['annotations']):
        image_id = annotation['image_id']
        dataset[image_id]['annotations' ].append({'segmentation':annotation['segmentation'],
                                                  'iscrowd': annotation['iscrowd'],
                                                  'bbox': annotation['bbox'],
                                                  'category_id': annotation['category_id']})
    print('Writing database...')
    for _, value in tqdm.tqdm(dataset.items()):
        sample_name = value['file_name']
        filename, _ = os.path.splitext(sample_name)
        # Serializing json 
        json_object = json.dumps(value, indent = 4)
        # Writing to sample.json
        with open(os.path.join(args.output,filename+'.json'), "w") as outfile:
            outfile.write(json_object)
    
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

