# -*- coding: utf-8 -*-
import torch as t
import numpy as np
import torch.utils.data as Data
from data.util import read_image
import utils.array_tool as at
from gensim.models import Word2Vec
import nltk
from data.dataset import Transform
import time
import json
from collections import defaultdict
import itertools
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-']
filter_word = english_punctuations + nltk.corpus.stopwords.words('english')

# param_map = {"image_record": "/data/yangy/coco/select_20coco_imgs.pkl",
#              "image_dir": {'train':"/data/yangy/coco/images/train/",
#                            'val':"/data/yangy/coco/images/val/"},
#              "text_file": "/data/yangy/coco/select_20coco_text.npy",
#              "label_file": "/data/yangy/coco/select_20coco_label.npy"}


dataset_names = ["MS-COCO"
                 # , "NUS-WIDE", "FLICKR", "IAPR"
                 ]

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCODataset(Data.Dataset):
    def __init__(self, embedding_file,opt,pretrain=False):
        super(COCODataset, self).__init__()
        dataType = 'train2017'
        self.param_map = opt.param_map
        annDir = opt.param_map['ann_dir']
        ann_caption_file = '{}/captions_{}.json'.format(annDir, dataType)
        ann_instance_file = '{}/instances_{}.json'.format(annDir, dataType)
        self.n = opt.n
        self.train_coco_caption = COCO(ann_caption_file)
        self.train_coco_instance = COCO(ann_instance_file)
        cat_ids = self.train_coco_instance.getCatIds()
        self.n_class = len(cat_ids)
        self.label_to_index = dict(zip(cat_ids,range(self.n_class)))
        self.train_img_ids = np.array(self.train_coco_caption.getImgIds())
        self.pretrain = pretrain

        dataType = 'val2017'
        ann_caption_file = '{}/captions_{}.json'.format(annDir, dataType)
        ann_instance_file = '{}/instances_{}.json'.format(annDir, dataType)
        self.val_coco_caption = COCO(ann_caption_file)
        self.val_coco_instance = COCO(ann_instance_file)

        self.val_img_ids = np.array(self.val_coco_caption.getImgIds())

        self.proportion = int(len(self.train_img_ids) / len(self.val_img_ids))

        '''
        upset data.
        '''
        permutation = np.random.permutation(len(self.train_img_ids))
        self.train_img_ids = self.train_img_ids[permutation]

        permutation = np.random.permutation(len(self.val_img_ids))
        self.val_img_ids = self.val_img_ids[permutation]

        self.word_embedding = Word2Vec.load(embedding_file)
        self.vector_size = self.word_embedding.vector_size
        self.tsf = Transform(opt.min_size, opt.max_size)


    def get_test_data(self):
        n = len(self.val_img_ids)
        val_imgs = []
        val_texts = t.zeros((n, self.n, self.vector_size))
        val_bboxes = []
        val_labels = []
        for i in range(n):
            image, text, bbox, labels = self._get_data_from_img_id(self.train_img_ids[i], 'train')
            val_imgs.append(image)
            val_texts[i, :text.shape[0], :] = text
            val_bboxes.append(bbox)
            val_labels.append(labels)
        return val_imgs,val_texts,val_bboxes,val_labels

    def __getitem__(self, index):
        if self.pretrain == False:
            val_id = self.val_img_ids[index % len(self.val_img_ids)]
            val_image, val_text, val_bbox, val_labels = self._get_data_from_img_id(val_id, 'val')
            val_txt = t.zeros((1, self.n, self.vector_size))
            val_txt[0, :val_text.shape[0], :] = val_text
            train_imgs = []
            train_texts = t.zeros((self.proportion, self.n, self.vector_size))
            train_bboxes = []
            train_labels = []
            for i in range(index * self.proportion, (index + 1) * self.proportion):
                image, text, bbox, labels = self._get_data_from_img_id(self.train_img_ids[i], 'train')
                train_imgs.append(image)
                train_texts[i - index * self.proportion, :text.shape[0], :] = text
                train_bboxes.append(bbox)
                train_labels.append(labels)

            return (train_imgs, val_image), (train_texts, val_txt), train_bboxes, train_labels
        else:
            image, text, bbox, label = self._get_data_from_img_id(self.train_img_ids[index], 'train')
            image = at.tonumpy(image)
            print('org image',image)
            bbox = at.tonumpy(bbox)
            label = at.tonumpy(label)
            img, bbox, label, scale = self.tsf((image, bbox, label))
            print('img',img)
            # TODO: check whose stride is negative to fix this instead copy all
            # some of the strides of a given numpy array are negative.
            return img.copy(), bbox.copy(),text , label.copy(), scale


    def _get_data_from_img_id(self, img_id, type):
        if type == 'train':
            coco_caption = self.train_coco_caption
            coco_instance = self.train_coco_instance
        else:
            coco_caption = self.val_coco_caption
            coco_instance = self.val_coco_instance

        image = read_image(self.param_map['image_dir'][type] + coco_caption.imgs[img_id]['file_name'])
        image = at.totensor(image).float()
        ann_id = coco_caption.getAnnIds(img_id)
        txt = coco_caption.loadAnns(ann_id)[0]['caption']
        corpus = [word for word in nltk.tokenize.word_tokenize(txt.lower()) if word not in filter_word]
        corpus_vector = t.from_numpy(
            np.array([self.word_embedding[word] for word in corpus if word in self.word_embedding.wv.vocab]))

        instance_id = coco_instance.getAnnIds(img_id)
        ins = coco_instance.loadAnns(instance_id)
        bbox = t.FloatTensor([i['bbox'] for i in ins])
        labels = t.IntTensor([self.label_to_index[i['category_id']] for i in ins])

        return image, corpus_vector, bbox, labels

    def __len__(self):
        return int(len(self.train_img_ids) / self.proportion)

class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]




