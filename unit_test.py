import numpy as np
import sys
import os
import pickle

def centroid_rands(centroid_rand_init, language, speaker_id):
    herman_centroid_rands = np.load("./data/kamperetal_init_centroids/"
                                    +language+"/"+speaker_id+"_rand.npy").transpose()

    if(np.allclose(herman_centroid_rands, centroid_rand_init, atol=0.001)):
        print("UNIT TEST PASSED: RANDOM CENTROIDS FOR SAMPLING ARE THE SAME AS KAMPER ET AL")
        print("\tTOTAL DIFF: "+str(np.sum(herman_centroid_rands - centroid_rand_init)))
    else:
        print("UNIT TEST FAILED: RANDOM CENTROIDS FOR SAMPLING ARE NOT THE SAME AS KAMPER ET AL")
        print("\tTOTAL DIFF: "+str(np.sum(herman_centroid_rands - centroid_rand_init)))
        sys.exit()

def segments_and_transcriptions(segments_and_transcipts_ours, language, speaker, epoch):

    with open(os.path.join('./data/kamperetal_segmentation/',
                           language,
                           "epoch_"+str(epoch),
                           speaker+'.pkl'),"rb") as f:
        segments_kamper_etal = pickle.load(f)

    for key, kamper_segment in segments_kamper_etal.items():
        our_segment =  [el[1]for el in segments_and_transcipts_ours[key]]
        our_segment = sorted(our_segment)
        kamper_segment = sorted(kamper_segment)

        if (our_segment != kamper_segment):
            print("UNIT TEST FAILED: SEGMENTS IN  EPOCH "+str(epoch)+" FROM UTT "+key+" ARE DIFFERENT AS KAMPER ET AL")
            print("OURS: "+str(our_segment))
            print("KAMPERS: "+str(kamper_segment))
            sys.exit()

    print("UNIT TEST PASSED: SEGMENTS IN  EPOCH "+str(epoch)+" ARE THE SAME AS KAMPER ET AL")

    with open(os.path.join('./data/kamperetal_transcripts/',
                           language,
                           "epoch_"+str(epoch),
                           speaker+'.pkl'),"rb") as f:
        transcripts_kamper_etal = pickle.load(f)

    for key, kamper_transcripts in transcripts_kamper_etal.items():
        our_transcripts =  [el[0]for el in segments_and_transcipts_ours[key]]
        if(our_transcripts != kamper_transcripts):
            print("OURS: "+str(our_transcripts))
            print("KAMPER: "+str(kamper_transcripts))
            print("UNIT TEST FAILED: TRANSCRIPTS IN  EPOCH "+str(epoch)+" FROM UTT "+key+" ARE DIFFERENT AS KAMPER ET AL")
            sys.exit()

    print("UNIT TEST PASSED: TRANSCRIPTS IN  EPOCH "+str(epoch)+" ARE THE SAME AS KAMPER ET AL")

def centroids(centroids_ours, language, speaker, epoch):

    centroids_kamperetal = np.load(os.path.join('./data/kamperetal_epochs_centroids/',
                                                language,
                                                "epoch_"+str(epoch),
                                                speaker+'.npy'))

    if (np.allclose(centroids_kamperetal, centroids_ours, atol=0.001)):
        print("UNIT TEST PASSED: CENTROIDS EPOCH "+str(epoch)+" ARE THE SAME AS KAMPER ET AL")
        print("\tTOTAL DIFF: " + str(np.sum(centroids_kamperetal - centroids_ours)))
    else:
        print("UNIT FAILED: CENTROIDS EPOCH "+str(epoch)+" ARE DIFFERENT AS KAMPER ET AL")
        print("\tTOTAL DIFF: " + str(np.sum(centroids_kamperetal - centroids_ours)))
        sys.exit()

def initial_centroids_and_segments(centroids, initial_segments, language, speaker_id):

    centroid_kampereral = np.load('./data/kamperetal_init_centroids/'+language+'/'+speaker_id+'.npy')

    if(np.allclose(centroid_kampereral, centroids, atol=0.001)):
        print("UNIT TEST PASSED: INITIAL CENTROIDS ARE THE SAME AS KAMPER ET AL")
        print("\tTOTAL DIFF: "+str(np.sum(centroid_kampereral - centroids)))

        with open('./data/kamperetal_init_segments/'+language+'/'+speaker_id+'_init_segs.pkl', 'rb') as f:
            initial_segments_kamperetal = pickle.load(f)

            if set(initial_segments_kamperetal.keys()) == set(initial_segments.keys()):
                for key in initial_segments_kamperetal.keys():

                    our_initial_segment = [ el[1] for el in initial_segments[key] ]
                    if initial_segments_kamperetal[key] != our_initial_segment:
                        print(f'The value of key {key} is different in each segmentation')
                        sys.exit()
            else:
                print("UNIT TEST FAILED: KEYS IN INITIAL SEGMENTATION ARE NOT THE SAME")
                sys.exit()
            print("UNIT TEST PASSED: INITIAL SEGMENTATION IS THE SAME AS KAMPER ET AL")
    else:
        print("UNIT TEST FAILED: CENTROIDS ARE NOT THE SAME AS KAMPER ET AL")
        print("\tTOTAL DIFF: "+str(np.sum(centroid_kampereral - centroids)))
        sys.exit()

def utterance_ids(feat_ours, language, speaker):

        main_path = os.path.join("/disk/scratch1/ramons/data/zerospeech_seg/mfcc_herman/",language,speaker)
        mfcc_herman_ids = sorted(list(np.load(os.path.join(main_path, 'raw_mfcc.npz')).keys()))

        if(sorted(list(feat_ours.keys())) != mfcc_herman_ids):
            print("the IDs (name and numbers) of the vad are NOT the same as in herman's mfccs")
            print(sorted(list(feat_ours.keys())))
            print(mfcc_herman_ids)
            sys.exit()
        else:
            print("the IDs (name and numbers) of the vad are the same as in herman's mfccs")
