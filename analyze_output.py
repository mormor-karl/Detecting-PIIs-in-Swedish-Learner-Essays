import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

#######################################################################

def split_tags_and_tokens(input: list):
    '''A function that splits every entry in a list by whitespace and into two separate lists.
    
    Args:
        input (list): A list where every entry is a string containing whitespace.
        
    Returns:
        Two lists of lists, containing the first and the second element of every entry from the original list, split by samples.
    '''
    tokens = []
    tags = []

    temp_token = []
    temp_tag = []
    
    for line in input:
        if len(line.strip()) > 1:
            token = line.strip().split()[0]
            tag = ' '.join(line.strip().split()[1:])

            temp_token.append(token)
            temp_tag.append(tag)

        else:  # if it's a break
            tokens.append(temp_token)
            tags.append(temp_tag)
            # reset
            temp_token = []
            temp_tag = []

    return tokens, tags

def get_tags_and_tokens(filename: str):
    '''A function that extracts tokens and the corresponding tags from a .txt file.
    
    Args:
        filename (str): The name of the file.
    
    Returns:
        A list of tokens and a list of tags.
    '''
    with open(filename) as f:
        tags = f.readlines()
        tokens, tags = split_tags_and_tokens(tags)
        
    return tokens, tags

def get_metadata(filename: str):
    '''A function that extracts the metadata from a .txt file.
    
    Args:
        filename (str): The name of the file.
    
    Returns:
        A list of tokens, a list of essay ids, and a list of original tags.
    '''
    with open(filename) as f:
        tags = f.readlines()
        tokens, other = split_tags_and_tokens(tags)
        
        essay_ids = []
        original_tags = []
        for doc in other:
            temp_essay_ids = []
            temp_original_tags = []
            for entry in doc:
                if ' ' in entry:
                    temp_essay_ids.append(entry.split()[0])
                    temp_original_tags.append(entry.split()[1])
                else:
                    temp_essay_ids.append(entry)
                    temp_original_tags.append(' ')
            essay_ids.append(temp_essay_ids)
            original_tags.append(temp_original_tags)
    return tokens, essay_ids, original_tags

def get_measures(gold_standard: list, predictions: list, path: str, model_name: str, labels: list = [], matrix: bool = False, details: bool = False):
    '''A function intended for retrieving a selection of evaluation measures for comparing the gold standard and the tagger
    annotations. The measures are printed out and include accuracy, Matthew's Correlation Coefficient, per-class precision 
    and recall, as well as a confusion matrix, which, in addition, get saved locally. These measures are calculated using 
    functions from sklearn and pyplot.
    
    Args:
        gold_standard (list[str]): A list of gold standard labels.
        predictions (list[str]): A list of predicted labels.
        path (str): The path for saving the outputs.
        model_name (str): The name of the evaluated model.
        labels (list[str]): A list of labels (if it needs to be specified).
        matrix (bool): Whether or not to produce a confusion matrix.
    '''
    
    if isinstance(gold_standard[0], list):
        gold_standard_list = [x for sentence in gold_standard for x in sentence]
    if isinstance(predictions[0], list):
        predictions_list = [x for sentence in predictions for x in sentence]

    if labels == []:  # setting up a list of labels based on the training data
        labels = sorted(list(set([x for sentence in gold_standard for x in sentence])))
    else:
        if isinstance(labels[0], list):
            labels = [x for sentence in labels for x in sentence]

    # writng out the measures
    with open(path + model_name + '_results.txt', 'w') as f:
        f.write('MEASURES:\n')
        f.write(f'Accuracy: {"{:.2%}".format(sklearn.metrics.accuracy_score(gold_standard_list, predictions_list))}\n')
        f.write(f'Precision (weighted): {"{:.2%}".format(sklearn.metrics.precision_score(gold_standard_list, predictions_list, average="weighted", zero_division=0))}\n')
        f.write(f'Recall (weighted): {"{:.2%}".format(sklearn.metrics.recall_score(gold_standard_list, predictions_list, average="weighted", zero_division=0))}\n')
        f.write(f'F1 (weighted): {"{:.2%}".format(sklearn.metrics.f1_score(gold_standard_list, predictions_list, average="weighted", zero_division=0))}\n')
        f.write(f'Matthew\'s Correlation Coefficient: {"{:.2%}".format(sklearn.metrics.matthews_corrcoef(gold_standard_list, predictions_list))}\n')
        if details:
            f.write('\n')
            f.write('MEASURES PER CLASS:\n')
            precision = sklearn.metrics.precision_score(gold_standard_list, predictions_list, average=None, labels=labels, zero_division=0)
            weighted_precision = ((precision[0]*1142) + (precision[1]*86)) / (1142 + 86)  # manually change weights
            f.write('Precision:\n')
            for i in range(0,len(labels)):
                f.write(f'\t{labels[i]}: {"{:.2%}".format(precision[i])}\n')
            f.write(f'\tWeighted precision for B and I: {"{:.2%}".format(weighted_precision)}\n')
            recall = sklearn.metrics.recall_score(gold_standard_list, predictions_list, average=None, labels=labels, zero_division=0)
            weighted_recall = ((recall[0]*1142) + (recall[1]*86)) / (1142 + 86)
            f.write('Recall:\n')
            for i in range(0,len(labels)):
                f.write(f'\t{labels[i]}: {"{:.2%}".format(recall[i])}\n')
            f.write(f'\tWeighted recall for B and I: {"{:.2%}".format(weighted_recall)}\n')
            f1 = sklearn.metrics.f1_score(gold_standard_list, predictions_list, average=None, labels=labels, zero_division=0)
            weighted_f1 = ((f1[0]*1142) + (f1[1]*86)) / (1142 + 86)
            f.write('F1:\n')
            for i in range(0,len(labels)):
                f.write(f'\t{labels[i]}: {"{:.2%}".format(f1[i])}\n')
            f.write(f'\tWeighted F1 for B and I: {"{:.2%}".format(weighted_f1)}\n')
    
    # printing out and saving the confusion matrix
    if matrix:
        # print('Confusion matrix:')
        sns.set_context('paper', font_scale=1.5)
        cm = sklearn.metrics.confusion_matrix(gold_standard_list, predictions_list, normalize='true', labels=labels)  # recall 
        ax = sns.heatmap(cm, cmap=sns.color_palette('cividis'), annot=True, xticklabels=labels, yticklabels=labels, cbar=False, linewidths=0.1, linecolor='white')
        ax.set(xlabel='Predicted tag', ylabel='True tag')
        plt.yticks(rotation=0)
        # matrix = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
        # fig, ax = plt.subplots(figsize=(12,12))
        # matrix.plot(ax=ax)
        
        plt.savefig(path + model_name + "_confusion_matrix.jpg", bbox_inches='tight')
        
def get_comparison(standard: list, predictions: list, tokens: list, essay_ids: list, original_tags: list, errors_only: bool = True):
    '''A function that returns a comparison of where mistakes were made during annotation.
    
    Args:
        standard (list): A list of gold standard annotations.
        predictions (list): A list of predicted annotations.
        tokens (list): A list of original tokens corresponding to the tags.
        essay_ids (list): A list of corresponding essay IDs.
        original_tags (list): A list of original tags corresponding to the tokens.
    
    Returns:
        A Pandas dataframe containing the mismatched annotations, their context, tokens, and other data.
    '''

    if errors_only: 
        problematic = []
        for j, sample in enumerate(predictions):
            for i, ann in enumerate(sample):
                if standard[j][i] != ann:
                    if i >= 5:
                        preceding = tokens[j][i-5:i]
                    elif i != 0:
                        preceding = tokens[j][:i]
                    else:
                        preceding = ''
                        
                    if i != len(tokens[j])-5:
                        succeeding = tokens[j][i+1:i+6]
                    elif i != len(tokens[j]):
                        succeeding = tokens[j][i+1:]
                    else:
                        succeeding = ''
                    
                    problematic.append((essay_ids[j][i], tokens[j][i], ' '.join([' '.join(preceding), tokens[j][i], ' '.join(succeeding)]), standard[j][i], predictions[j][i], original_tags[j][i]))
           
        problematic_frame = pd.DataFrame(problematic, columns=['Essay ID', 'Token', 'Context', 'Gold Standard', 'Prediction', 'Original Tag'])
        
        return problematic_frame

    else:
        all_examples = []
        for j, sample in enumerate(predictions):
            for i, ann in enumerate(sample):
                if i >= 5:
                    preceding = tokens[j][i-5:i]
                elif i != 0:
                    preceding = tokens[j][:i]
                else:
                    preceding = ''
                        
                if i != len(tokens[j])-5:
                    succeeding = tokens[j][i+1:i+6]
                elif i != len(tokens[j]):
                    succeeding = tokens[j][i+1:]
                else:
                    succeeding = ''
                    
                all_examples.append((essay_ids[j][i], tokens[j][i], ' '.join([' '.join(preceding), tokens[j][i], ' '.join(succeeding)]), standard[j][i], predictions[j][i], original_tags[j][i]))
           
        frame = pd.DataFrame(all_examples, columns=['Essay ID', 'Token', 'Context', 'Gold Standard', 'Prediction', 'Original Tag'])
        
        return frame
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='The path for saving the output.')
    parser.add_argument('model_name', help='The name of the model.')
    
    parser.add_argument('--bert_path', required=False, default='./bert/', help='The path to the folder with one or more folders with BERT runs.')
    parser.add_argument('--data_path', required=False, default='./data/', help='The path to the folder with test data.')
  
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    
    print('Loading in the data...')
    
    predictions = args.bert_path + args.model_name + '/test_predictions.txt'
    standard = args.data_path + 'test.txt'
    metadata = args.data_path + 'meta_test.txt'
    
    tokens, predicted_tags = get_tags_and_tokens(predictions)
    _, standard_tags = get_tags_and_tokens(standard)
    _, essay_ids, original_tags = get_metadata(metadata)
    
    print('Calculating measures...')
    
    get_measures(standard_tags, predicted_tags, args.path, args.model_name, details=True, matrix=True)

    print('Generating comparisons...')
    
    comparison = get_comparison(standard_tags, predicted_tags, tokens, essay_ids, original_tags).sort_values('Gold Standard').reset_index(drop=True)
    comparison.to_excel(args.path + args.model_name + '_errors.xlsx')
    
    comparison = get_comparison(standard_tags, predicted_tags, tokens, essay_ids, original_tags, errors_only=False).sort_values('Gold Standard').reset_index(drop=True)
    comparison.to_excel(args.path + args.model_name + '_all.xlsx')
    
    print('Done!')
    