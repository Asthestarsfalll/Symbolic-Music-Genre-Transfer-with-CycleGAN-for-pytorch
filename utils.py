import os.path
import torch.distributed as dist
import numpy as np
import pretty_midi


def load_npy(npy_dir):
    npy = np.load(npy_dir).astype(np.float32)
    return npy


def save_midis(bars, file_path, tempo=80.0):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])),
                                  bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))),
                                 axis=2)

    padded_bars = padded_bars.reshape(-1, 64,
                                      padded_bars.shape[2], padded_bars.shape[3])
    padded_bars_list = []
    for ch_idx in range(padded_bars.shape[3]):
        padded_bars_list.append(padded_bars[:, :, :, ch_idx].reshape(padded_bars.shape[0],
                                                                     padded_bars.shape[1],
                                                                     padded_bars.shape[2]))
    # this is for multi-track version
    # write_midi.write_piano_rolls_to_midi(padded_bars_list, program_nums=[33, 0, 25, 49, 0],
    #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)

    # this is for single-track version
    write_piano_rolls_to_midi(piano_rolls=padded_bars_list,
                              program_nums=[0],
                              is_drum=[False],
                              filename=file_path,
                              tempo=tempo,
                              beat_resolution=4)


def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=16):
    # Calculate time per pixel
    tpp = 60.0 / tempo / float(beat_resolution)
    threshold = 60.0 / tempo / 4
    phrase_end_time = 60.0 / tempo * 4 * piano_roll.shape[0]
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape(
        (piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    piano_roll_diff = np.concatenate(
        (np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
    # Iterate through all possible(128) pitches

    for note_num in range(128):
        # Search for notes
        start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
        start_time = list(tpp * (start_idx[0].astype(float)))

        end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
        end_time = list(tpp * (end_idx[0].astype(float)))

        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]

        temp_start_time = [i for i in start_time]
        temp_end_time = [i for i in end_time]

        for i in range(len(start_time)):
            # print(start_time)
            if start_time[i] in temp_start_time and i != len(start_time) - 1:
                t = []
                current_idx = temp_start_time.index(start_time[i])
                for j in range(current_idx + 1, len(temp_start_time)):
                    if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                        t.append(j)
                for _ in t:
                    temp_start_time.pop(t[0])
                    temp_end_time.pop(t[0])

        start_time = temp_start_time
        end_time = temp_end_time
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]

        if len(end_time) < len(start_time):
            d = len(start_time) - len(end_time)
            start_time = start_time[:-d]
        # Iterate through all the searched notes
        for idx in range(len(start_time)):
            if duration[idx] >= threshold:
                # Create an Note object with corresponding note number, start time and end time
                note = pretty_midi.Note(
                    velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
                # Add the note to the Instrument object
                instrument.notes.append(note)
            else:
                if start_time[idx] + threshold <= phrase_end_time:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                            end=start_time[idx] + threshold)
                else:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                            end=phrase_end_time)
                # Add the note to the Instrument object
                instrument.notes.append(note)
    # Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)


def write_piano_roll_to_midi(piano_roll, filename, program_num=0, is_drum=False, velocity=100,
                             tempo=120.0, beat_resolution=16):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Create an Instrument object
    instrument = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
    # Set the piano roll to the Instrument object
    set_piano_roll_to_instrument(
        piano_roll, instrument, velocity, tempo, beat_resolution)
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)


def write_piano_rolls_to_midi(piano_rolls, program_nums=None, is_drum=None, filename='test.mid', velocity=100,
                              tempo=120.0, beat_resolution=24):
    if len(piano_rolls) != len(program_nums) or len(piano_rolls) != len(is_drum):
        print("Error: piano_rolls and program_nums have different sizes...")
        return False
    if not program_nums:
        program_nums = [0, 0, 0]
    if not is_drum:
        is_drum = [False, False, False]
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Iterate through all the input instruments
    for idx in range(len(piano_rolls)):
        # Create an Instrument object
        instrument = pretty_midi.Instrument(
            program=program_nums[idx], is_drum=is_drum[idx])
        # Set the piano roll to the Instrument object
        set_piano_roll_to_instrument(
            piano_rolls[idx], instrument, velocity, tempo, beat_resolution)
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)


def to_binary(bars, threshold=0.0):
    track_is_max = np.equal(bars,np.max(bars,axis=1,keepdims=True))
    track_pass_threshold = (bars > threshold)
    out_track = np.logical_and(track_is_max, track_pass_threshold)
    return out_track

def check_path(path, rank=0):
    if rank == 0 and not os.path.exists(path):
        os.mkdir(path)


if __name__=='__main__':
    test_dir = './test/test.mid'
    m = pretty_midi.PrettyMIDI(test_dir)

    # a=m.instruments[0].get_piano_roll()
    # print(a)
    n = m.instruments[0].notes
    for i in n:
        print(i.get_duration())

