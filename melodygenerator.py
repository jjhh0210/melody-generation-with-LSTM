import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="model.h5"):


        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)  #load dictionary

        self._start_symbols = ["/"] * SEQUENCE_LENGTH  #delimiter = start symbol로 설정


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the DL model and returns a midi file."""

        #1. create seed with start symbols (initiate generation with this seed melody)
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed -> int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]  #1st 확률분포 get
            # [0.1, 0.2, 0.1, 0.6] -> 합쳐서 1

            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we are at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilites, temperature):
        """
        Samples an index from a probability array reapplying softmax using temperature
        # predictions (nd.array): Array containing probabilities for each of the possible outputs.
        # temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))    #softmax 적용하여 확률분포 remodel

        choices = range(len(probabilites)) # ex) [0, 1, 2, 3] -> 이중에서 sample 1개 choice
        index = np.random.choice(choices, p=probabilites)   #each item of choices - 대응하는 probability

        return index    #return selected output symbol


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """
        Converts a melody into a MIDI file
        # min_duration : Duration of each time step in quarter length
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):


                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case "_" : 연장,지속의 의미
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed1 = "60 _ 60 _ 67 _ 67 _ 69 _ 69 _ 67 _ _ _ 65 _ 65 _ 64 _ 64 _ 62 _ 62 _ 60 _ _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    seed3 = "64 _ 69 _ _ _ 71 _ 72 _ _ 71 69 _ 76 _"
    seed4 = "64 _ 62 _ 60 _ 60 _ _ _ _ 55 _ 55 _ 60 _ 62 _ 62 _ _ _ _"
    melody = mg.generate_melody(seed4, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    mg.save_melody(melody)


