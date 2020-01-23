"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC, LinearSVC
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import gen_batches
from tqdm import tqdm

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                if args.mode == 'TRAIN':
                    dataset_tmp = facenet.get_dataset(args.data_dir, max_per_class=args.max_nrof_images_per_class)
                    train_set, _ = split_dataset(dataset_tmp, args.min_nrof_images_per_class,
                                                 args.nrof_train_images_per_class)
                    dataset = train_set
                elif args.mode == 'CLASSIFY':
                    dataset_tmp = facenet.get_dataset(args.data_dir, max_per_class=None)
                    _, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class,
                                                args.nrof_train_images_per_class)
                    dataset = test_set
            else:
                if args.mode == 'TRAIN':
                    dataset = facenet.get_dataset(args.data_dir, max_per_class=args.max_nrof_images_per_class)
                elif args.mode == 'CLASSIFY':
                    dataset = facenet.get_dataset(args.data_dir, max_per_class=None)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset'

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print(paths[:10])
            print(labels[:10])
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            clf_dir, _ = os.path.split(classifier_filename_exp)
            data_dir, dataset_name = os.path.split(args.data_dir)
            if not dataset_name:
                _, dataset_name = os.path.split(data_dir)
            emb_file = os.path.join(clf_dir, dataset_name + '_embeddings.pkl')

            if os.path.exists(emb_file):
                print('Loading features from file:', emb_file)
                with open(emb_file, 'rb') as infile:
                    emb_array = pickle.load(infile)
            else:
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in tqdm(range(nrof_batches_per_epoch)):
                    start_index = i*args.batch_size
                    end_index = min((i+1)*args.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, args.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

                with open(emb_file, 'wb') as outfile:
                    pickle.dump(emb_array, outfile)

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                # model = SVC(kernel='linear', probability=True)
                model = LinearSVC(class_weight='balanced')
                model.fit(emb_array, labels)
            
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                if args.n_jobs > 1:
                    # Using parallelized prediction
                    n_jobs = int(args.n_jobs)
                    X = emb_array
                    n_samples, n_features = X.shape
                    batch_size = n_samples // n_jobs

                    def _predict(method, X, sl):
                        return method(X[sl])

                    results = Parallel(n_jobs)(delayed(_predict)(model.predict, X, sl)
                                               for sl in gen_batches(n_samples, batch_size))

                    # predictions = np.zeros((nrof_images, len(class_names)))
                    predictions = np.zeros((nrof_images,))
                    start = 0
                    for result in results:
                        print(result.shape)
                        predictions[start:start+result.shape[0]] = result
                        start += result.shape[0]
                else:
                    predictions = model.predict(emb_array)

                # best_class_indices = np.argmax(predictions, axis=1)
                # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                best_class_indices = predictions

                submission = [(os.path.basename(image), class_names[int(pred)]) for image, pred in zip(paths, best_class_indices)]

                preds_csv = os.path.join(clf_dir, args.classifier_filename.replace('.pkl', '') + '.csv')
                df_submission = pd.DataFrame(submission)
                print('Saving submission to:', preds_csv)
                print(df_submission.head())
                df_submission.to_csv(preds_csv, header=False, index=False)
                # with open(preds_csv, 'w') as fp:
                #    writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)
                #    writer.writerows(df_submission.values)

                # for i in range(len(best_class_indices)):
                #    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--max_nrof_images_per_class', type=int,
                        help='Maximum number of images per class', default=100)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    parser.add_argument('--n_jobs', type=int, help='Number of parallel jobs used during inference', default=1)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
