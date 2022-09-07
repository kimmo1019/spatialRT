import yaml,time
import argparse
from spatialRT import *
tf.keras.utils.set_random_seed(123)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as f:
        params = yaml.load(f)
    #model = scDEC(params)
    model = spatialRT(params)
    #model = spatialRT_v2(params)
    model.train()