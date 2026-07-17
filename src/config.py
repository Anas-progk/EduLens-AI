"""
Central configuration for the student engagement classification system.
Edit this file to tune hyperparameters -- all other scripts import from here.

v2 changes:
  - drop_rate: 0.25 -> 0.30  (stronger regularisation for small dataset)
  - drop_path_rate: 0.10 -> 0.15
  - train_csv: confirmed as merged_train.csv (NOT the aggressive synthetic version)
"""
import torch

CONFIG = {
    # -- Data paths
    'raw_daisee_dir'   : 'data/raw/daisee',
    'frames_dir'       : 'data/processed/daisee/frames',
    'labels_csv'       : 'data/processed/daisee/labels.csv',
    # IMPORTANT: Use merged_train.csv (original data, no aggressive synthetic).
    # The aggressive final_merged_train.csv (from run 5) hurt NE_F1 by 15%.
    # Only switch to merged_train_light.csv AFTER confirming baseline improves.
    'train_csv'        : 'data/splits/merged_train.csv',
    'val_csv'          : 'data/splits/custom_val.csv',
    'test_csv'         : 'data/splits/custom_test.csv',

    # -- Frame extraction
    'frame_stride'     : 15,
    'max_frames'       : 8,

    # -- Model
    'model_name'       : 'swin_tiny_patch4_window7_224',
    'num_classes'      : 2,
    'image_size'       : 224,
    # Dropout increased in v2 (was 0.25 / 0.10).
    # Swin-Tiny on ~1000 clips needs stronger regularisation to prevent
    # the backbone from memorising training identities.
    'drop_rate'        : 0.30,
    'drop_path_rate'   : 0.15,

    # -- DataLoader (frame-level)
    'batch_size'       : 32,
    'num_workers'      : 2,
    'pin_memory'       : True,

    # -- Three-Phase Training (v4 frame-level -- kept for reference)
    'phase1_epochs'     : 10,
    'phase1_lr_head'    : 3e-4,
    'phase1_wd'         : 1e-2,

    'phase2_epochs'     : 12,
    'phase2_lr_backbone': 5e-6,
    'phase2_lr_head'    : 2e-5,
    'phase2_wd'         : 1e-2,
    'phase2_warmup'     : 2,

    'phase3_epochs'     : 8,
    'phase3_lr_backbone': 2e-6,
    'phase3_lr_head'    : 5e-6,
    'phase3_wd'         : 1e-2,

    # -- Regularisation
    'grad_clip'         : 0.3,
    'eval_threshold'    : 0.35,

    # -- Early stopping (global fallback; per-phase patience set in PHASES dict)
    'patience'          : 8,
    'min_delta'         : 0.001,

    # -- Clip-level model (temporal, v5)
    'clip_batch_size'   : 4,
    'n_frames_clip'     : 8,

    # -- Checkpointing
    'save_dir'          : '/content/drive/MyDrive/engagement_weights',
    'best_model_name'   : 'best_clip_model.pth',

    # -- Engagement label threshold (DAiSEE)
    'engagement_threshold': 2,

    # -- Device
    'device'            : 'cuda' if torch.cuda.is_available() else 'cpu',

    # -- Custom dataset (classroom videos recorded by user)
    'custom_raw_dir'    : 'custom_dataset/raw_videos',
    'custom_proc_dir'   : 'custom_dataset/processed',
    'custom_meta_csv'   : 'custom_dataset/tracking_metadata.csv',
    'custom_clips_csv'  : 'custom_dataset/clips_catalog.csv',
    'custom_annot_csv'  : 'custom_dataset/annotations.csv',
    'custom_train_csv'  : 'data/splits/custom_train.csv',
    'custom_val_csv'    : 'data/splits/custom_val.csv',
    'custom_test_csv'   : 'data/splits/custom_test.csv',
    'merged_train_csv'  : 'data/splits/merged_train.csv',
    'merged_val_csv'    : 'data/splits/merged_val.csv',
    'merged_test_csv'   : 'data/splits/merged_test.csv',

    # -- EduAction supplementary dataset
    # EduAction clips are already cropped 224x224 with one person per clip.
    # Labels from folder name: EduAction_E -> Engaged, EduAction_NE -> Not Engaged.
    'edu_e_dir'         : 'custom_dataset/EduAction_E',
    'edu_ne_dir'        : 'custom_dataset/EduAction_NE',
    'edu_proc_dir'      : 'custom_dataset/processed_edu',
    'edu_frames_csv'    : 'custom_dataset/eduaction_frames.csv',

    # -- Light synthetic NE (generate_ne_light.py output)
    # Use merged_train_light.csv only AFTER confirming baseline on merged_train.csv.
    'merged_train_light_csv': 'data/splits/merged_train_light.csv',
}
