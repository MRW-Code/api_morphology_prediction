import argparse


parser = argparse.ArgumentParser(usage='python main.py')

parser.add_argument('-m', '--model', action='store', dest='model',
                  default='RF', choices=['RF', 'ResNet'])

parser.add_argument('--kfold', action='store_true', dest='kfold')

parser.add_argument('-d', '--dataset', action='store', dest='dataset',
                  default='matt', choices=['laura', 'matt'])

parser.add_argument('--calc_desc', action='store_false', dest='load_data')

parser.add_argument('--deep', action='store_true', dest='is_deep')

args = parser.parse_args()
