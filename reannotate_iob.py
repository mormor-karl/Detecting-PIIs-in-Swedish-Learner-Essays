import sys
import os
import random
import math
import argparse
from sklearn.utils.class_weight import compute_class_weight

parent_dir = os.path.dirname('../src/')  # so that we can access utils.py
sys.path.append(parent_dir)

from utils import *

#######################################################################

def extract_pii_annotation(document_list: list, max_len: int | bool=200):
    '''A function that returns a list of IOB-annotated graphs based on the SweLL pilot files extracted with the functions available in 
    utils.py.
    
    Args:
        document_list (list): A list of SweLL file contents as retrieved by functions from utils.py.
        max_len (int | bool): The maximum length for one text / graph. No splitting if 0.

    Returns:
        A list of modified Svala-graph sources, now including IOB tags to represent sensitive data/passages.
        A list of document IDs for the reannotated documents.
    '''
    annotated_graphs = []
    doc_ids = [document['id'] for document in document_list]
    
    for i, document in enumerate(document_list):
        graph = align_pii_annotation(document['svala_graph'])
        
        for entry in graph:
            entry['doc_id'] = doc_ids[i]
        
        if max_len == 0:
            annotated_graphs.append(graph)
        else:
            # splitting up the data
            n_chunks = math.floor(len(graph) / max_len)
            if n_chunks < 0:
                annotated_graphs.append(graph)
            else:
                for i in range(n_chunks):
                    new_graph = graph[(i-1)*max_len : i*max_len]
                    if len(new_graph) > 0:
                        annotated_graphs.append(new_graph)
                try:        
                    new_graph = graph[n_chunks*max_len:]
                    if len(new_graph) > 0:
                        annotated_graphs.append(new_graph)
                except IndexError:
                    pass

    return annotated_graphs, doc_ids

def align_pii_annotation(graph: dict):
    '''A function that adds the IOB tags to Svala source annotation.
    
    Args:
        graph (dict): A single Svala-graph for a document.

    Returns:
        A list of dicts for every token in the source, including IOB annotation.
    '''
    source = graph['source']
    edges = graph['edges']

    annotated_source = []

    iob_annotations, original_labels = reannotate_2_iob(edges)
    
    # the assert statements help identify where an error comes from (but none should pop up now)
    assert len(iob_annotations) == len(source)

    for i, annotation in enumerate(iob_annotations):
        id, tag = annotation
        original_id, original_tag = original_labels[i]

        source_dict = source[i]
        assert source_dict['id'] == id
        assert id == original_id
            
        source_dict['tag'] = tag
        source_dict['original_tag'] = original_tag

        annotated_source.append(source_dict)
        
    return annotated_source

def reannotate_2_iob(edges: dict):
    '''A function that takes the edges from a Svala-graph and creates IOB annotations based on them.
    
    Args:
        edges (dict): The edges part of a Svala-graph.

    Returns:
        A list token-id and IOB annotation pairs.
    '''
    iob_annotations = []
    original_labels = []
    
    for edge in edges.values():
        source_ids = [x for x in edge['ids'] if x.startswith('s')]
        # selecting the appropriate annotation from among I, O, and B
        if len(source_ids) == 1 and len(edge['labels']) == 0:  # no annotation
            iob_annotations.append((source_ids[0], 'O'))
            original_labels.append((source_ids[0], ''))
        elif len(source_ids) == 1:  # and len(edge['labels']) > 0 - there is some annotation 
            iob_annotations.append((source_ids[0], 'B'))
            original_labels.append((source_ids[0], edge['labels'][0]))
        else:  # len(source_ids) > 1:
            new_source_ids = [int(x.strip('s')) for x in source_ids]
            new_source_ids.sort()  # this is needed due to a quirk in the annotation (some ranges show up in the wrong order)
            iob_annotations.append(('s' + str(new_source_ids[0]), 'B'))
            original_labels.append(('s' + str(new_source_ids[0]), edge['labels'][0]))
            for i in range(1, len(new_source_ids)):
                iob_annotations.append(('s' + str(new_source_ids[i]), 'I'))
                original_labels.append(('s' + str(new_source_ids[i]), edge['labels'][0]))
    return iob_annotations, original_labels

def produce_bert_files(
    annotated_graphs: list, 
    seed: int=25, 
    test_size: float=0.1, 
    dev_size: float=0.1, 
    filenames: list=['train.txt.tmp', 'test.txt.tmp', 'dev.txt.tmp'], 
    meta_filenames: list=['meta_train.txt', 'meta_test.txt', 'meta_dev.txt'],
    path: str='./'
    ):
    '''A function that shuffles, splits, and writes out the data in the desired format.
    
    Args:
        annotated_graphs (list): A list of IOB-reannotated graphs.
        seed (int): The random seed to be used.
        test_size (float): The size of the test set, expressed using a decimal fraction.
        dev_size (float): The size of the dev set, expressed using a decimal fraction.
        filenames (list): A list of filenames for the train, test, and dev sets.
        path (str): The path to save the files at.
    '''
    #print(annotated_graphs[0])
    data = balanced_shuffle_and_split(annotated_graphs, seed=seed, test_size=test_size, dev_size=dev_size)
    
    if not os.path.exists(path):
        os.mkdir(path)

    for i, name in enumerate(filenames):
        write_2_file(data[i], path + name, path + meta_filenames[i])
        
    print()
    print(f'The counts of instances of the B, I, and O classes are: {count_classes(data[0] + data[1] + data[2])}')
    print()
    print(f'Weights for the B, I, and O classes: {calculate_class_weights(data[0] + data[1] + data[2])}')
    print()

    print(f'Data printed into {[path + filename for filename in filenames]} and {[path + filename for filename in meta_filenames]}') 
    
def balanced_shuffle_and_split(
    annotated_graphs: list, 
    seed: int=25, 
    test_size: float=0.1, 
    dev_size: float=0.1
    ):
    '''A function that shuffles, splits, and writes out the data in the desired format.
    
    Args:
        annotated_graphs (list): A list of IOB-reannotated graphs.
        seed (int): The random seed to be used.
        test_size (float): The size of the test set, expressed using a decimal fraction.
        dev_size (float): The size of the dev set, expressed using a decimal fraction.
        
    Returns:
        Balanced test, train, and dev sets.
    '''
    random.seed(seed)

    # splitting up the data into graphs with IB and graphs without it so that each set has an equal number of them
    with_iobs = []
    without_iobs = []
    
    for graph in annotated_graphs:
        iobs = False
        for entry in graph:
            if entry['tag'] != 'O':
                iobs = True
        
        if iobs:
            with_iobs.append(graph)
        else:
            without_iobs.append(graph)
            
    random.shuffle(with_iobs)
    random.shuffle(without_iobs)
    
    # selecting the maximum length for the sets so that our data is balanced
    max_len = min([len(with_iobs), len(without_iobs)])
    
    # trimming the shuffled sets
    with_iobs = with_iobs[:max_len]
    without_iobs = without_iobs[:max_len]

    # selecting cutoff points
    test_cutoff = math.floor(test_size * max_len)
    dev_cutoff = math.floor((dev_size+test_size) * max_len)
    
    # recombining and reshuffling the data
    test_data = with_iobs[:test_cutoff] + without_iobs[:test_cutoff]
    dev_data = with_iobs[test_cutoff:dev_cutoff] + without_iobs[test_cutoff:dev_cutoff]
    train_data = with_iobs[dev_cutoff:] + without_iobs[dev_cutoff:]
    
    random.shuffle(test_data)
    random.shuffle(dev_data)
    random.shuffle(train_data)
    
    data = [train_data, test_data, dev_data]   
    return data
    
def write_2_file(data: list, path: str, path_meta: str):
    '''A function that writes the data in the desired format to a file.
    
    Args:
        data (list): A list of IOB-reannotated graphs representing a single subset (test, train, or dev).
        path (str): The path to save the files at, including filename.
    '''
    with open(path, 'w') as f:
        for graph in data:
            for entry in graph:
                token = entry['text']
                tag = entry['tag']
                f.write(token.strip('\n ') + ' ' + tag + '\n')
            f.write('\n')
            
    with open(path_meta, 'w') as f:
        for graph in data:
            for entry in graph:
                token = entry['text']
                original_tag = entry['original_tag']
                doc_id = entry['doc_id']
                f.write(token.strip('\n ') + ' ' + doc_id + ' ' + original_tag + '\n')
            f.write('\n')
            
def calculate_class_weights(annotated_graphs: list):
    '''A function that calculates class weights for the data.
    
    Args:
        annotated_graphs (list): A list of IOB-reannotated graphs.
        
    Returns:
        A vector of weights for the classes in the order of B, I, O.
    '''
    classes = ['B', 'I', 'O']
    y = [entry['tag'] for graph in annotated_graphs for entry in graph]
    
    return compute_class_weight(class_weight='balanced', classes=classes, y=y)

def count_classes(annotated_graphs: list):
    '''A function that counts instances of the classes in the data.
    
    Args:
        annotated_graphs (list): A list of IOB-reannotated graphs.
        
    Returns:
        A list of counts for the classes in the order of B, I, O.
    '''
    y = [entry['tag'] for graph in annotated_graphs for entry in graph]
    
    b = y.count('B')
    i = y.count('I')
    o = y.count('O')
    
    return [b, i, o]
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='The path to the folder containing the SweLL pilot data.')
    parser.add_argument('output_path', help='The path to the folder the output should be saved in.')
    
    parser.add_argument('--test_size', required=False, default=0.1, help='The required test set size as a decimal fraction. Default: 0.1')
    parser.add_argument('--dev_size', required=False, default=0.1, help='The required dev size as a decimal fraction. Default: 0.1')
    parser.add_argument('--seed', required=False, default=25, help='The desired random seed. Default: 25')
    parser.add_argument('--train_file', required=False, default='train.txt.tmp', help='The desired name for the train file. Default: train.txt')
    parser.add_argument('--test_file', required=False, default='test.txt.tmp', help='The desired name for the test file. Default: test.txt')
    parser.add_argument('--dev_file', required=False, default='dev.txt.tmp', help='The desired name for the dev file. Default: dev.txt')
    parser.add_argument('--max_len', required=False, default=200, help='The max length for graphs (longer ones get split up). Default: 200')
    
    
    
    args = parser.parse_args()
    
    filenames=[args.train_file, args.test_file, args.dev_file]
    document_list, _ =  read_swell_directory(args.input_path)
    print(f'Loaded in {len(document_list)} documents.')
    annotated_graphs, _ = extract_pii_annotation(document_list, max_len=int(args.max_len))
    produce_bert_files(
        annotated_graphs, 
        seed=int(args.seed), 
        test_size=float(args.test_size), 
        dev_size=float(args.dev_size), 
        filenames=filenames,
        path=args.output_path
        )
    