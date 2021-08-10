import os
import music21 as m21
import json
import numpy as np
import tensorflow.keras as keras

KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "datasets"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64
ACCEPTABLE_DURATIONS=[
    0.25,   #16th note  = 1/4 beat
    0.5,    #8th note = 1/2 beat
    0.75,   #16th note+8th note 점8분음표
    1.0,    #quarter note = 1 beatW
    1.5,    #dotted quarter note 점4분음표
    2,      #half note = 2 beats
    3,      #three quarter note 점2분음표
    4       #whole note = 4 beats

]

def load_songs_in_kern(dataset_path):

    songs = []

    #go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        #only krn 파일만 load
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path,file)) #song = music21 score(stream)
                songs.append(song)

    return songs


def has_acceptable_durations(song,acceptable_durations):

    #song 내의 모든 notes 하나씩 check - note 하나라도 non acceptable duration 가질시 해당 song filter
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):
    #1. get key from the song (a score > parts > measures > notes(key))
    parts = song.getElementsByClass(m21.stream.Part)    #get all parts in score
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)    #get all measures in parts
    key = measures_part0[0][4]      #where the key is stored!(key의 위치)

    #1-2. estimate key using music21!
    if not isinstance(key,m21.key.Key):
        key = song.analyze("key")   #위에서 key가 추출되지 않을 시 music21가 song 분석하여 key를 반환

    #3. get interval for transposition ex) B maj --> C maj 하려면 B 와 C 사이의 간격(거리)를 구하여 변경해야함
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch("C"))    #key.tonic 과  C major 간 간격
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))   #key.tonic 과 A minor 간 간격

    #4. transpose song by 계산된 interval
    transposed_song = song.transpose(interval)

    return transposed_song

def encode_song(song, time_step = 0.25):
    #ex) pitch = 60(MIDI), duration = 1.0 -->[60,"_","_","_"] 즉, 1 time step = 1/4박자(16분음표)  , '_' means holding

    encoded_song = [] #결국 time series 표현법으로 인코딩된 노래의 모든 음표, 쉼표가 리스트 형태로 반환됨

    for event in song.flat.notesAndRests:

        #handel notes(음표)
        if isinstance(event,m21.note.Note):
            symbol = event.pitch.midi #60

        #handel rests(쉼표)
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        #convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    encoded_song = " ".join(map(str, encoded_song)) #all elements to string

    return encoded_song


def preprocess(dataset_path):

    #1. load the folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):

        # 2. filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song,ACCEPTABLE_DURATIONS):
            continue

        # 3. transpose songs to C maj / A min
        song = transpose(song)

        # 4. encode songs with music time series representation
        encoded_song = encode_song(song)

        # 5. save encoded songs to text file
        save_path = os.path.join(SAVE_DIR,str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)

        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path,file_dataset_path, sequence_length):

    new_song_delimiter = "/ "* sequence_length  #song간의 구분자(end of the song)
    songs = ""

    #load encoded songs and add delimiters
    #visit all datasets in folder
    for path,_,files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter  #모든 encoded songs의 strings --> 1 string으로 결합

    songs = songs[:-1]

    #save string that contains all dataset
    with open(file_dataset_path,"w") as fp:
        fp.write(songs)

    return songs

def create_mapping(songs, mapping_path):

    mappings = {}

    #identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    #create mapping(integer 사전 제작)
    for i,symbol in enumerate(vocabulary):
        mappings[symbol] = i

    #save vocabulary to a json file (key:symbol,value:int)
    with open(mapping_path, "w") as fp:
        json.dump(mappings,fp,indent = 4)

def convert_songs_to_int(songs):
    int_songs = []

    #load mappings(dict)
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    #cast songs string to a list
    songs = songs.split()

    #map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_sequences(sequence_length):
    # ex) [11,12,13,14, ... ] -> i: [11,12] , t:13 ; i:[12,13] , t:14

    # load songs & map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # generate the training sequences(inputs, targets 생성)
    # 100 symbols, 64 sequence length --> 생성할 수 있는 sequences 수 : 100-64 = 36
    inputs = []
    targets= []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences
    # inputs dim: (num of sequences, sequence length,vocabulary size)
    # ex) [[0,1,2] [1,1,2]] -->[[[1,0,0],[0,1,0],[0,0,1]],[]]
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs,num_classes = vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()

