import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    # requiredNamed
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--config', type=str, default='config/homo_export_labels_angiogram.yaml', help="The config file for training or exporting the label")
    return parser.parse_args()