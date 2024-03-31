import torch
import logging
from typing import List, Tuple
import os
import sys
import pretty_midi
from dataclasses import dataclass
from IPython import display
from torch.utils.data import IterableDataset

DEFAULT_INSTRUMENT = "piano"

# logging.basicConfig(level=logging.ERROR)
_log = logging.getLogger()
_log.setLevel(logging.ERROR)
_log.addHandler(logging.StreamHandler(sys.stdout))


def debug_mode_on():
    _log.setLevel(logging.DEBUG)


@dataclass
class Note:
    # instrument_name: str = ''
    pitch: int = 0
    vel: int = 0
    duration: float = 0.0
    start: float = 0.0
    end: float = 0.0
    step: float = 0.0


SAMPLING_RATE = 16_000
NUM_MIDI_CHANS = 16  # 0..15


class MidiFile(IterableDataset):
    def __init__(
        self, midi_file: pretty_midi.PrettyMIDI, sequence_len: int = 3, name: str = ""
    ):
        self.midi_file = midi_file

        assert sequence_len > 2, "The sequence length must be >= 2"
        self.sequence_len = sequence_len
        self.name = name

        self._notes = None
        self._tensor = None
        self._file_duration = 0  # In seconds

    @classmethod
    def from_filepath(cls, file_path: str, sequence_len: int = 3):
        try:
            midi_file = pretty_midi.PrettyMIDI(file_path)
        except Exception as e:
            _log.error(f"Fail to read {file_path}: {e}")
            return None

        return cls(midi_file, sequence_len, name=file_path)

    # @classmethod
    # def from_messages(cls, midi_msgs: List[mido.Message], sequence_len: int = 3):
    #     return cls(mido.MidiFile(tracks=[mido.MidiTrack(midi_msgs)]), sequence_len)

    def instrument_names(self) -> List[str]:
        return [
            pretty_midi.program_to_instrument_name(i.program)
            for i in self.midi_file.instruments
        ]

    def has_instrument(self, instrument_name) -> bool:
        instrument_name = instrument_name.lower()
        return any(map(lambda i: instrument_name in i.lower(), self.instrument_names()))

    # instrument_name lets you select which instrument to select.
    def notes(self) -> List[Note]:
        # TODO: uncomment once finished experimenting.
        if self._notes is not None:
            return self._notes

        instrument_names = enumerate(self.instrument_names())

        # TODO: change it
        instrument_name = DEFAULT_INSTRUMENT

        if instrument_name != "":
            instrument_names = filter(
                lambda i: instrument_name.lower() in i[1].lower(), instrument_names
            )

        notes: List[pretty_midi.Note] = []
        for idx, _ in instrument_names:
            notes.extend(self.midi_file.instruments[idx].notes)

        if len(notes) == 0:
            _log.debug(f"No {instrument_name} notes found for {self.name}")
            return []

        notes = sorted(notes, key=lambda note: note.start)
        last_start = notes[0].start
        self._file_duration = notes[-1].end
        self._notes = []

        for n in notes:
            self._notes.append(
                Note(
                    pitch=n.pitch,
                    vel=n.velocity,
                    start=n.start,
                    end=n.end,
                    step=n.start - last_start,
                    duration=n.duration,
                )
            )

            last_start = n.start

        return self._notes

    def tensor(self) -> torch.Tensor:
        if self._tensor is not None:
            return self._tensor

        # TODO: remove the limit
        notes = self.notes()
        dtype = torch.float32

        pitches = torch.empty(len(notes), 1, dtype=dtype)
        velocities = torch.empty_like(pitches)
        steps = torch.empty_like(pitches)
        durations = torch.empty_like(pitches)

        for i, n in enumerate(notes):
            pitches[i] = n.pitch
            velocities[i] = n.vel
            steps[i] = n.step
            durations[i] = n.duration

        self._tensor = torch.cat((pitches, velocities, steps, durations), dim=1)

        return self._tensor

    def __iter__(self):
        self.p = -1  # A pointer idx for iterations.

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.p += 1
        t = self.tensor()

        if self.p + 1 + self.sequence_len > len(t):
            raise StopIteration

        return (
            t[self.p : self.p + self.sequence_len, :],
            t[self.p + 1 : self.p + 1 + self.sequence_len, :],
        )

    def display_audio(self, sampling_rate: int = SAMPLING_RATE) -> display.Audio:
        sampled_song = self.midi_file.fluidsynth(sampling_rate)
        return display.Audio(sampled_song, rate=sampling_rate)


def load_midi_files(*dir_paths: str) -> List[MidiFile]:
    file_paths = []
    for dir_path in dir_paths:
        for file_name in os.listdir(dir_path):
            file_paths.append(os.path.join(dir_path, file_name))

    return [
        midi_file
        for path in file_paths
        if (midi_file := MidiFile.from_filepath(path)) is not None
    ]


def sequences(t: torch.Tensor, sequence_len: int) -> torch.Tensor:
    if sequence_len + 1 > len(t):
        raise ValueError(
            f"Sequence length ({sequence_len}) is greater than the tensor length ({len(t)})"
        )

    seqs = []
    for i in range(len(t) - sequence_len):
        seqs.append((t[i : i + sequence_len, :], t[i + sequence_len]))

    return seqs


# def sequence_len_view(t: torch.Tensor, sequence_len: int) -> torch.Tensor:
#     if sequence_len == 1:
#         return t

#     # Pud with zeroes if dims don't match.
#     if (rem := t.shape[0] % sequence_len) != 0:
#         t = torch.cat((t, torch.zeros((sequence_len - rem, t.shape[1]))))

#     r_size = int(t.shape[0] / sequence_len)
#     c_size = t.shape[1] * sequence_len

#     return t.view((r_size, c_size))


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
