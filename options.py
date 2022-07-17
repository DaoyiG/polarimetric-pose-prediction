import os
import argparse

file_dir = os.path.dirname(__file__)


class PolarPoseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PolarPose options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="")

        self.parser.add_argument("--object_id2name_path",
                                 type=str,
                                 help="path to the file containing meta information of object id and corresponding name",
                                 default="")

        self.parser.add_argument("--object_ply_path",
                                 type=str,
                                 help="path to the ply file of the object",
                                 default="")

        self.parser.add_argument("--object_metainfo_path",
                                 type=str,
                                 help="path to the file containing meta information of object bbox and diameter, and symmetry information",
                                 default="")


        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="log")

        self.parser.add_argument("--train_split",
                                 type=str,
                                 help="file of train split",
                                 default='')

        self.parser.add_argument("--val_split",
                                 type=str,
                                 help="file of train split",
                                 default='')

        self.parser.add_argument("--test_split",
                                 type=str,
                                 help="file of train split",
                                 default='')

        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="which resnet backbone to use",
                                 default=34)

        self.parser.add_argument("--use_skips",
                                 type=bool,
                                 help="whether to skip extracted features from encoder to decoder",
                                 default=True)
        self.parser.add_argument("--pretrained",
                                 type=bool,
                                 help="whether to use pretrained encoder backbone",
                                 default=True)
        self.parser.add_argument("--pretrained_backbone",
                                 type=str,
                                 help="which pretrained backbone to use",
                                 default="torchvision://resnet34")


        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="model")
        self.parser.add_argument("--device",
                                 type=str,
                                 help="cuda or cpu",
                                 default="cuda")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=2e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=200)
        self.parser.add_argument("--milestones",
                                 type=list,
                                 help="optimizer milestones",
                                 default=[50, 100, 150])
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["polarposenet"])


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options