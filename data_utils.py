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
def load_dataset(dataset, speaker, feature_type, layer=10, vad="prevad"):

    main_path = os.path.join("/disk/scratch1/ramons/data/zerospeech_seg/mfcc_herman/",dataset,speaker)

    landmark_file = open(os.path.join(main_path, 'landmarks.pkl'), 'rb')
    landmark = pickle.load(landmark_file)
    landmark_file.close()   

    landmarks_aux = {}
    for key in sorted(landmark):
        landmarks_aux[key] = landmark[key]
    landmarks_dict = landmarks_aux

    if(feature_type == "mfcc"):
        feat_np = np.load(os.path.join(main_path, 'raw_mfcc.npz'))
        return landmarks_dict, feat_np

    elif("hubert" in feature_type):

        mfcc_herman_ids = sorted(list(np.load(os.path.join(main_path, 'raw_mfcc.npz')).keys()))

        main_path = os.path.join("/disk/scratch1/ramons/data/hubert_data/seg/zsc/",
                                 feature_type,str(layer),
                                 "norm",
                                 dataset,
                                 "postvad")

        feat_np = np.load(os.path.join(main_path, speaker+"_features_frame.npz"))

        if(sorted(list(feat_np.keys())) != mfcc_herman_ids):
            print("the IDs (name and numbers) of the vad are NOT the same as in herman's mfccs")
            print(sorted(list(feat_np.keys())))
            print(mfcc_herman_ids)
            sys.exit()
        else:
            print("the IDs (name and numbers) of the vad are the same as in herman's mfccs")

        return landmarks_dict, feat_np
    else:
        print("format not supported")
        sys.exit()
