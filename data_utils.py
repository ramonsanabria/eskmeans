import os
import sys
import pickle
import numpy as np


#TODO sanity check dataset
def load_dataset(dataset, speaker, format):

    main_path = os.path.join("/disk/scratch1/ramons/data/zerospeech_seg/mfcc_herman/",dataset,speaker)

    landmark_file = open(os.path.join(main_path, 'landmarks.pkl'), 'rb')
    landmark = pickle.load(landmark_file)
    landmark_file.close()   

    landmarks_aux = {}
    for key in sorted(landmark):
        landmarks_aux[key] = landmark[key]
    landmarks_dict = landmarks_aux

    if(format == "npz"):
        feat_np = np.load(os.path.join(main_path, 'raw_mfcc.npz'))
        return landmarks_dict, feat_np
    elif(format == "scp"):
        feat_scp_path = os.path.join(main_path, 'raw_mfcc.scp')
        return landmarks_dict, feat_scp_path
    else:
        print("format not supported")
        sys.exit()
    #feat_scp_file = open(os.path.join(main_path, 'raw_mfcc.scp'))

    #feat_scp_dict={}
    #for line in feat_scp_file.readlines():
    #    feat_scp_dict[line.split()[0]] = line.strip()

    # feat_scp = []

    # for key in sorted(feat_scp_dict):
    #
    #     parts_scp = feat_scp_dict[key].split()
    #     utt_id = parts_scp[0]
    #     file, shift = parts_scp[1].split(':')
    #
    #     feat_scp.append((utt_id, file, int(shift)))
    #     #landmark_ordered.append((landmark[utt_id], utt_id))
    #
    #return landmark_ordered, feat_scp, feat_scp_file

