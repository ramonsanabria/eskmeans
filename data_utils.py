import os
import sys
import pickle
import numpy as np
import unit_test
from collections import defaultdict
import socket

def write_ramons(unsup_landmarks, unsup_transcript, speaker_id, output_folder):

    class_dict= defaultdict(list)
    print("WRITING RESULTS AT: "+str(os.path.join(output_folder,speaker_id+".tdev")))

    for key in unsup_transcript.keys():
        for idx, class_id in enumerate(unsup_transcript[key]):
            start = float(key.split("_")[-1].split("-")[0])/100
            new_key = "_".join(key.split("_")[:-1])

            final_start = start+float(unsup_landmarks[key][idx][0]/100)
            final_end = start+float(unsup_landmarks[key][idx][1]/100)
            class_dict[class_id].append((final_start,final_end,new_key))

    with open(os.path.join(output_folder,speaker_id+".tdev"), "w") as result_file:
        for class_id in class_dict.keys():
            result_file.write("Class "+str(class_id)+"\n")
            for segment in class_dict[class_id]:
                result_file.write(str(segment[2])+" "+str(segment[0])+" "+str(segment[1])+"\n")
            result_file.write("\n")

def filter_short_segments(landmarks_dict, feat_np, minimum_duration):
    deleted_segments = 0
    new_feat_np = {}
    new_landmarks_dict = {}

    for key in list(landmarks_dict.keys()):
        start, end = key.split("_")[-1].split("-")
        if((int(end) - int(start)) < minimum_duration):
            deleted_segments += 1
        else:
            new_feat_np[key] = feat_np[key]
            new_landmarks_dict[key] = landmarks_dict[key]

    print(str(deleted_segments)+" deleted segments due to duration < "+str(minimum_duration))
    return new_landmarks_dict, new_feat_np

#TODO sanity check dataset
def load_dataset(language,
                 speaker,
                 feature_type,
                 minimum_duration,
                 unit_test_flag,
                 feature_layer="10",
                 vad_position="prevad"):

    if socket.gethostname() == "banff.inf.ed.ac.uk":
        main_path_base = os.path.join("/disk/scratch_fast/ramons/data/zerospeech_seg/mfcc_herman/",language,speaker)
    else:
        main_path_base = os.path.join("/disk/scratch1/ramons/data/zerospeech_seg/mfcc_herman/",language,speaker)

    landmark_file = open(os.path.join(main_path_base, 'landmarks.pkl'), 'rb')
    landmark = pickle.load(landmark_file)
    landmark_file.close()   

    landmarks_aux = {}
    for key in sorted(landmark):
        landmarks_aux[key] = landmark[key]
    landmarks_dict = landmarks_aux

    if(feature_type == "mfcc"):
        feat_np = np.load(os.path.join(main_path_base, 'raw_mfcc.npz'))
        return filter_short_segments(landmarks_dict, feat_np, minimum_duration)

    elif("hubert" in feature_type):


        if socket.gethostname() == "banff.inf.ed.ac.uk":
            main_path = os.path.join("/disk/scratch_fast/ramons/data/hubert_data/seg/zsc/",
                                     feature_type,
                                     str(feature_layer),
                                     language,
                                     vad_position)
        else:
            main_path = os.path.join("/disk/scratch1/ramons/data/hubert_data/seg/zsc/",
                                     feature_type,
                                     str(feature_layer),
                                     language,
                                     vad_position)

        feat_np = np.load(os.path.join(main_path, speaker+"_features_frame.npz"))

        if(unit_test_flag):
            unit_test.utterance_ids(feat_np, language, speaker)


        return filter_short_segments(landmarks_dict, feat_np, minimum_duration)
    else:
        print("format not supported")
        sys.exit()
