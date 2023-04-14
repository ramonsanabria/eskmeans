import os
import sys
import pickle
import numpy as np
import unit_test
from collections import defaultdict

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

#TODO sanity check dataset
def load_dataset(language, speaker, feature_type, unit_test_flag, layer=10):

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
        return landmarks_dict, feat_np

    elif("hubert" in feature_type):


        main_path = os.path.join("/disk/scratch1/ramons/data/hubert_data/seg/zsc/",
                                 feature_type, str(layer),
                                 "norm",
                                 language,
                                 "postvad")

        feat_np = np.load(os.path.join(main_path, speaker+"_features_frame.npz"))

        unit_test.utterance_ids(feat_np, language, speaker)

        return landmarks_dict, feat_np
    else:
        print("format not supported")
        sys.exit()
