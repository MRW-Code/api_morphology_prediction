import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from src.utils import args
from src.preprocess import get_ml_df, get_lab_data
from src.images import images_from_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fastai.vision.all import *

if __name__ == '__main__':
    print(args)

if args.dset == 'descriptor':
    print('Using Descriptor Dataset')

    ## ADD is binary call in utils

    df = get_ml_df(args.binary)
    import models.models
    model = getattr(models.models, args.model)
    labels = np.array(df.loc[:, ['eye_morphology']]).reshape(-1)
    features = df[df.columns[~df.columns.isin(['SMILES', 'eye_morphology'])]]

    if args.kfold:
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42, )
        score = cross_val_score(model, features, labels, cv=kf, scoring='accuracy')
        print(f'Scores for each fold are: {score}')
        print(f'Average score: {"{:.2f}".format(score.mean())}')

    else:
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            labels,
                                                            random_state=0,
                                                            test_size=0.2)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(score)

        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        forest_importances = pd.Series(result.importances_mean, index=features.columns)
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importance using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()

elif args.dset == 'image':
    print('Using Image Dataset')
    df = get_lab_data(args.binary)
    if args.load_data:
        print('Images loaded from folder')
    else:
        images_from_dataset(df)
        print('Images generated and saved to file')

    path = './data/images'
    fnames = get_image_files(path)
    def label_func(x) : return x.parent
    tfms = aug_transforms(pad_mode='zeros', mult=2, batch=True)
    dls = ImageDataLoaders.from_path_func(path, fnames, label_func, bs=64, batch_tfms=tfms, seed=0)
    learn = cnn_learner(dls, resnet18, pretrained=True, metrics=[accuracy])
    learn.fit(5)
    learn.unfreeze()
    learn.fit(10)



