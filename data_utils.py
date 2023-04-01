import os
import sys
import pickle
import numpy as np
import pathlib
from collections import defaultdict

def write_ramons(unsup_landmarks, unsup_transcript, dataset, speaker_id):

    pathlib.Path(os.path.join("./results",dataset)).mkdir(parents=True, exist_ok=True)
    #get classes
    class_dict= defaultdict(list)
    print("WRITING RESULTS AT: "+str(os.path.join("./results",dataset)))

    all_names = ["_".join(el.split("_")[:-1]) for el in unsup_transcript.keys()]
    #if(len(set(all_names)) > 1):
    #    print("ERROR: we are processing more than one speaker "+str(set(all_names)))
    #    sys.exit()


    for key in unsup_transcript.keys():
        for idx, class_id in enumerate(unsup_transcript[key]):
            start = float(key.split("_")[-1].split("-")[0])/100
            new_key = "_".join(key.split("_")[:-1])

            final_start = start+float(unsup_landmarks[key][idx][0]/100)
            final_end = start+float(unsup_landmarks[key][idx][1]/100)
            if(final_start == 193.31):
                print(key)
                print(unsup_landmarks[key][idx])

            class_dict[class_id].append((final_start,final_end,new_key))

    with open(os.path.join("./results",dataset,speaker_id+".tdev"), "w") as result_file:
        for class_id in class_dict.keys():
            result_file.write("Class "+str(class_id)+"\n")
            for segment in class_dict[class_id]:
                result_file.write(str(segment[2])+" "+str(segment[0])+" "+str(segment[1])+"\n")
            result_file.write("\n")




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

