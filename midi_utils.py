import torch, torch.nn.functional as F
import logging
from typing import List, Tuple
import os
import mido
from dataclasses import dataclass

mido.set_backend('mido.backends.rtmidi')

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

@dataclass
class Note:
    type: str = ''
    pitch: int = 0 
    vel: int = 0
    time: int = 0
    chan: int = 0 

NOTE_TYPES = { "note_on": 0, "note_off":1 }
NUM_MIDI_CHANS = 16 # 0..15

class MidiFile:
    def __init__(self, midi_file: mido.MidiFile, sequence_len = 3):
        self.midi_file = midi_file

        if sequence_len is None:
            sequence_len = 3

        assert sequence_len > 2, "The sequence length must be >= 2"
        self.sequence_len = sequence_len

        self._notes = None
        self._tensor = None

    @classmethod
    def from_filepath(cls, file_path: str, sequence_len = None):
        try:
            midi_file = mido.MidiFile(file_path)
        except Exception as e:
            logger.error(f"Fail to read {file_path}: {e}")
            return None

        return cls(midi_file, sequence_len)

    @classmethod
    def from_messages(cls, midi_msgs: List[mido.Message], sequence_len=None):
        return cls(mido.MidiFile(tracks=[mido.MidiTrack(midi_msgs)]), sequence_len)

    def messages(self) -> List[mido.Message]:
        return [msg for track in self.midi_file.tracks for msg in track]

    def notes(self)-> List[Note]:
        if self._notes is not None:
            return self._notes

        msgs = filter(lambda msg: msg.type in ["note_on", "note_off"], self.messages())
        self._notes = list(map(
            lambda msg: Note(
                    type = msg.type,
                    pitch=msg.note, 
                    chan=msg.channel, 
                    vel=msg.velocity,
                    time= msg.time,
            ), msgs,
        ))

        return self._notes
    
    def tensor(self) -> torch.Tensor:
        if self._tensor is not None:
            return self._tensor

        # TODO: remove the limit
        notes = self.notes()[:100]
        dtype = torch.float32

        num_types = len(NOTE_TYPES)
        one_hot_types = torch.empty((0, num_types), dtype=dtype)
        one_hot_chans = torch.empty((0,NUM_MIDI_CHANS), dtype=dtype)

        vels = torch.empty(len(notes), 1, dtype=dtype)
        pitches = torch.empty_like(vels)
        times = torch.empty_like(vels)

        for i, n in enumerate(notes):
            ht = F.one_hot(torch.tensor(NOTE_TYPES[n.type]), num_classes= num_types).unsqueeze(0)
            one_hot_types = torch.cat((one_hot_types, ht))

            hc = F.one_hot(torch.tensor(n.chan), num_classes=NUM_MIDI_CHANS).unsqueeze(0)
            one_hot_chans = torch.cat((one_hot_chans, hc))
            vels[i] = n.vel
            pitches[i]= n.pitch
            times[i]= n.time

        self._tensor = torch.cat((one_hot_types, one_hot_chans, vels, pitches, times), dim=1)

        return self._tensor

    def __iter__(self):
        self.p = -1 # A pointer idx for iterations.

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.p += 1
        t = self.tensor()

        if self.p + 1 + self.sequence_len + 1 >= len(t):
            raise StopIteration
         
        return t[self.p: self.p + self.sequence_len, :], t[self.p+1: self.p+1 + self.sequence_len, :]
        

    def play(self):
        port_name = ""
        for output_name in mido.get_output_names():
            if output_name.split(" ")[0].lower() == "fluid":
                port_name = output_name
                break

        assert port_name != "", "port name must not be empty"

        with mido.open_output(name=port_name) as outport:
            for msg in self.midi_file.play():
                outport.send(msg)


def load_midi_files(*dir_paths: str) -> List[MidiFile]:
    file_paths = []
    for dir_path in dir_paths:
        for file_name in os.listdir(dir_path):
            file_paths.append(os.path.join(dir_path, file_name))

    midi_files = []
    for path in file_paths:
        midi_file = MidiFile.from_filepath(path)
        if midi_file is not None:
            midi_files.append(midi_file)
    
    return midi_files


def sequence_len_view(t : torch.Tensor, sequence_len: int) -> torch.Tensor:
    if sequence_len == 1:
        return t

    # Pud with zeroes if dims don't match.
    if (rem := t.shape[0] % sequence_len) != 0:
        t = torch.cat((t, torch.zeros((sequence_len - rem, t.shape[1])))) 

    r_size = int(t.shape[0] / sequence_len)
    c_size = t.shape[1] * sequence_len 

    return t.view((r_size, c_size))
# def new_midi_file(notes: List[int], times: List[int], path_to_save: str = None) -> mido.MidiFile:
#     # flake8: noqa E509

#     assert len(notes) == len(times)

#     track = mido.MidiTrack()
#     velocity = 109

#     for note, time in zip(notes, times):
#         on = mido.Message('note_on', note=note, velocity=velocity, time=0)
#         off = mido.Message('note_off', note=note, velocity=0, time=time)
#         track.append(on)
#         track.append(off)

#     mid = mido.MidiFile(tracks=[track])

#     # Add the track to the MidiFile and save it
#     if path_to_save is not None:
#         mid.save(path_to_save)

#     return mid


