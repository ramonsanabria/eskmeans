import os
import sys
import pickle


#TODO sanity check dataset
def load_dataset(dataset, speaker):

    main_path = os.path.join("/disk/scratch1/ramons/data/zerospeech_seg/mfcc_herman/",dataset,speaker)
    landmarks = {}

    landmark_file = open(os.path.join(main_path, 'landmarks.pkl'), 'rb')
    landmark = pickle.load(landmark_file)
    landmark_file.close()

    feat_scp_file = open(os.path.join(main_path, 'mfcc.scp'))

    feat_scp = []
    landmark_ordered = []

    for line in feat_scp_file.readlines():
        parts = line.strip().split()

        utt_id = parts[0]
        file, shift = parts[1].split(':')

        feat_scp.append((utt_id, file, int(shift)))
        landmark_ordered.append((landmark[utt_id], utt_id))

    return landmark_ordered, feat_scp
