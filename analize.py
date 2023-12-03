from configs.main_config import *
from configs.argparse_config import parser, reminder

from all_models.masks.masks_module import analize_masks
from all_models.distance.video_sdd_distance import analize_distance
from all_models.full_module import vortex
from all_models.masks.masks_module import analize_masks
from configs.argparse_config import *
from configs.main_config import *

if __name__ == '__main__':
    args = parser.parse_args()
    mode, statistic = args.mode, args.statistic

    if mode == 'masks':
        analize_masks(
            retina_config_path=resnet_configuration,
            retina_weights_path=det_weights_path,
            mask_weight_path=cla_mask_path,
            video_path=videopath,
            statistic=statistic)

    elif mode == 'dist':
        analize_distance(
            sdd_config_path=yolo_config_path,
            sdd_weights_path=yolo_weights_path,
            video_path=videopath,
            statistic=statistic)

    elif mode == 'all':
        vortex(sdd_config_path=yolo_config_path,
               sdd_weights_path=yolo_weights_path,
               sdd_additional_config='configs/ssd_model',
               retina_config_path=resnet_configuration,
               retina_weights_path=det_weights_path,
               mask_weight_path=cla_mask_path,
               video_path=videopath,
               statistic=statistic)

    else:
        raise argparse.ArgumentTypeError(reminder)
