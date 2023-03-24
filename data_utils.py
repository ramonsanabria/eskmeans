import os
import sys
import pickle


#TODO sanity check dataset
def load_dataset(dataset, speaker):

    main_path = os.path.join("/disk/scratch1/ramons/data/zerospeech_seg/mfcc_herman/",dataset,speaker)

    landmark_file = open(os.path.join(main_path, 'landmarks.pkl'), 'rb')
    landmark = pickle.load(landmark_file)
    landmark_file.close()   

    landmarks_aux = {}
    for key in sorted(landmark):
        landmarks_aux[key] = landmark[key]
    landmark = landmarks_aux

    feat_scp_file = open(os.path.join(main_path, 'mfcc.scp'))

    feat_scp_dict={}
    for line in feat_scp_file.readlines():
        feat_scp_dict[line.split()[0]] = line.strip()

    feat_scp = []
    landmark_ordered = []

    for key in sorted(feat_scp_dict):

        parts_scp = feat_scp_dict[key].split()
        utt_id = parts_scp[0]
        file, shift = parts_scp[1].split(':')

        feat_scp.append((utt_id, file, int(shift)))
        landmark_ordered.append((landmark[utt_id], utt_id))

    return landmark_ordered, feat_scp
