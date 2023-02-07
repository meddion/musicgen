from typing import List
import os

import mido

mido.set_backend('mido.backends.rtmidi')


def notes_dataset(dir_path: str) -> tuple[List[int], List[mido.Message]]:
    mido_msgs = []
    note_values = []

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        midi_file = open_midi_file(file_path)
        notes, mido_msg = extract_notes(midi_file)
        mido_msgs.append(mido_msg)
        note_values.append(notes)

    return note_values, mido_msgs


def open_midi_file(path: str) -> mido.MidiFile:
    return mido.MidiFile(path)


def extract_notes(mid: mido.MidiFile) -> tuple[List[int], List[mido.Message]]:
    notes = []
    mido_msgs = []
    for track in mid.tracks:
        for msg in track:
            mido_msgs.append(msg)
            if msg.type == 'note_on':
                notes.append(msg.note)

    return notes, mido_msgs


def play(mid: mido.MidiFile):
    port_name = ""
    for output_name in mido.get_output_names():
        if output_name.split(" ")[0].lower() == "fluid":
            port_name = output_name
            break

    assert port_name != "", "port name must not be empty"

    with mido.open_output(name=port_name) as outport:
        for msg in mid.play():
            outport.send(msg)


def new_midi_note(note_value: int) -> dict[str, any]:
    return {'note': note_value, 'duration': 100, 'velocity': 109}


def new_midi_file(notes: List[int], path_to_save: str = None) -> mido.MidiFile:
    track = mido.MidiTrack()

    for note in map(new_midi_note, notes):
        on = mido.Message(
            'note_on', note=note['note'], velocity=note['velocity'], time=0)
        track.append(on)
        off = mido.Message(
            'note_off', note=note['note'], velocity=0, time=note['duration'])
        track.append(off)

    mid = mido.MidiFile(tracks=[track])

    # Add the track to the MidiFile and save it
    if path_to_save is not None:
        mid.save(path_to_save)

    return mid


def midi_file_from_msgs(midi_msgs: List[mido.Message]) -> mido.MidiFile:
    return mido.MidiFile(tracks=[mido.MidiTrack(midi_msgs)])
