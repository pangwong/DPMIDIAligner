from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import sys
import pretty_midi
import six
import math
import numpy as np

import util.music_pb2 as music_pb2


pretty_midi.pretty_midi.MAX_TICK = 1e10


class MIDIConversionError(Exception):
  pass


def midi_to_note_sequence(midi_data):
  """Convert MIDI file contents to a NoteSequence.

  Converts a MIDI file encoded as a string into a NoteSequence. Decoding errors
  are very common when working with large sets of MIDI files, so be sure to
  handle MIDIConversionError exceptions.

  Args:
    midi_data: A string containing the contents of a MIDI file or populated
        pretty_midi.PrettyMIDI object.

  Returns:
    A NoteSequence.

  Raises:
    MIDIConversionError: An improper MIDI mode was supplied.
  """
  # In practice many MIDI files cannot be decoded with pretty_midi. Catch all
  # errors here and try to log a meaningful message. So many different
  # exceptions are raised in pretty_midi.PrettyMidi that it is cumbersome to
  # catch them all only for the purpose of error logging.
  # pylint: disable=bare-except
  if isinstance(midi_data, pretty_midi.PrettyMIDI):
    midi = midi_data
  else:
    try:
      midi = pretty_midi.PrettyMIDI(six.BytesIO(midi_data))
    except:
      raise MIDIConversionError('Midi decoding error %s: %s' %
                                (sys.exc_info()[0], sys.exc_info()[1]))
  # pylint: enable=bare-except

  sequence = music_pb2.NoteSequence()

  # Populate header.
  sequence.ticks_per_quarter = midi.resolution
  sequence.source_info.parser = music_pb2.NoteSequence.SourceInfo.PRETTY_MIDI
  sequence.source_info.encoding_type = (
      music_pb2.NoteSequence.SourceInfo.MIDI)

  # Populate time signatures.
  for midi_time in midi.time_signature_changes:
    time_signature = sequence.time_signatures.add()
    time_signature.time = midi_time.time
    time_signature.numerator = midi_time.numerator
    try:
      # Denominator can be too large for int32.
      time_signature.denominator = midi_time.denominator
    except ValueError:
      raise MIDIConversionError('Invalid time signature denominator %d' %
                                midi_time.denominator)

  # Populate key signatures.
  for midi_key in midi.key_signature_changes:
    key_signature = sequence.key_signatures.add()
    key_signature.time = midi_key.time
    key_signature.key = midi_key.key_number % 12
    midi_mode = midi_key.key_number // 12
    if midi_mode == 0:
      key_signature.mode = key_signature.MAJOR
    elif midi_mode == 1:
      key_signature.mode = key_signature.MINOR
    else:
      raise MIDIConversionError('Invalid midi_mode %i' % midi_mode)

  # Populate tempo changes.
  tempo_times, tempo_qpms = midi.get_tempo_changes()
  for time_in_seconds, tempo_in_qpm in zip(tempo_times, tempo_qpms):
    tempo = sequence.tempos.add()
    tempo.time = time_in_seconds
    tempo.qpm = tempo_in_qpm

  # Populate notes by gathering them all from the midi's instruments.
  # Also set the sequence.total_time as the max end time in the notes.
  midi_notes = []
  midi_pitch_bends = []
  midi_control_changes = []
  for num_instrument, midi_instrument in enumerate(midi.instruments):
    for midi_note in midi_instrument.notes:
      if not sequence.total_time or midi_note.end > sequence.total_time:
        sequence.total_time = midi_note.end
      midi_notes.append((midi_instrument.program, num_instrument,
                         midi_instrument.is_drum, midi_note))
    for midi_pitch_bend in midi_instrument.pitch_bends:
      midi_pitch_bends.append(
          (midi_instrument.program, num_instrument,
           midi_instrument.is_drum, midi_pitch_bend))
    for midi_control_change in midi_instrument.control_changes:
      midi_control_changes.append(
          (midi_instrument.program, num_instrument,
           midi_instrument.is_drum, midi_control_change))

  for program, instrument, is_drum, midi_note in midi_notes:
    note = sequence.notes.add()
    note.instrument = instrument
    note.program = program
    note.start_time = midi_note.start
    note.end_time = midi_note.end
    note.pitch = midi_note.pitch
    note.velocity = midi_note.velocity
    note.is_drum = is_drum

  for program, instrument, is_drum, midi_pitch_bend in midi_pitch_bends:
    pitch_bend = sequence.pitch_bends.add()
    pitch_bend.instrument = instrument
    pitch_bend.program = program
    pitch_bend.time = midi_pitch_bend.time
    pitch_bend.bend = midi_pitch_bend.pitch
    pitch_bend.is_drum = is_drum

  for program, instrument, is_drum, midi_control_change in midi_control_changes:
    control_change = sequence.control_changes.add()
    control_change.instrument = instrument
    control_change.program = program
    control_change.time = midi_control_change.time
    control_change.control_number = midi_control_change.number
    control_change.control_value = midi_control_change.value
    control_change.is_drum = is_drum

  # TODO(douglaseck): Estimate note type (e.g. quarter note) and populate
  # note.numerator and note.denominator.

  return sequence

def midi_file_to_note_sequence(midi_file):
  """Converts MIDI file to a NoteSequence.

  Args:
    midi_file: A string path to a MIDI file.

  Returns:
    A NoteSequence.

  Raises:
    MIDIConversionError: Invalid midi_file.
  """
  with open(midi_file, 'rb') as f:
    midi_as_string = f.read()
    return midi_to_note_sequence(midi_as_string)


Pianoroll = collections.namedtuple(  
    'Pianoroll',
    ['active', 'weights', 'onsets', 'onset_velocities', 'active_velocities',
     'offsets', 'control_changes'])

MIN_MIDI_PITCH = 0  # Inclusive.
MAX_MIDI_PITCH = 127  # Inclusive.

# The amount to upweight note-on events vs note-off events.
ONSET_UPWEIGHT = 5.0

# The size of the frame extension for onset event.
# Frames in [onset_frame-ONSET_WINDOW, onset_frame+ONSET_WINDOW]
# are considered to contain onset events.
ONSET_WINDOW = 1


def sequence_to_pianoroll(
    sequence,
    frames_per_second,
    min_pitch,
    max_pitch,
    min_velocity=MIN_MIDI_PITCH,
    max_velocity=MAX_MIDI_PITCH,
    add_blank_frame_before_onset=False,
    onset_upweight=ONSET_UPWEIGHT,
    onset_window=ONSET_WINDOW,
    onset_length_ms=0,
    offset_length_ms=0,
    onset_mode='window',
    onset_delay_ms=0.0,
    min_frame_occupancy_for_label=0.0,
    onset_overlap=True):
  """Transforms a NoteSequence to a pianoroll assuming a single instrument.

  This function uses floating point internally and may return different results
  on different platforms or with different compiler settings or with
  different compilers.

  Args:
    sequence: The NoteSequence to convert.
    frames_per_second: How many frames per second.
    min_pitch: pitches in the sequence below this will be ignored.
    max_pitch: pitches in the sequence above this will be ignored.
    min_velocity: minimum velocity for the track, currently unused.
    max_velocity: maximum velocity for the track, not just the local sequence,
      used to globally normalize the velocities between [0, 1].
    add_blank_frame_before_onset: Always have a blank frame before onsets.
    onset_upweight: Factor by which to increase the weight assigned to onsets.
    onset_window: Fixed window size to activate around onsets in `onsets` and
      `onset_velocities`. Used only if `onset_mode` is 'window'.
    onset_length_ms: Length in milliseconds for the onset. Used only if
      onset_mode is 'length_ms'.
    offset_length_ms: Length in milliseconds for the offset. Used only if
      offset_mode is 'length_ms'.
    onset_mode: Either 'window', to use onset_window, or 'length_ms' to use
      onset_length_ms.
    onset_delay_ms: Number of milliseconds to delay the onset. Can be negative.
    min_frame_occupancy_for_label: floating point value in range [0, 1] a note
      must occupy at least this percentage of a frame, for the frame to be given
      a label with the note.
    onset_overlap: Whether or not the onsets overlap with the frames.

  Raises:
    ValueError: When an unknown onset_mode is supplied.

  Returns:
    active: Active note pianoroll as a 2D array..
    weights: Weights to be used when calculating loss against roll.
    onsets: An onset-only pianoroll as a 2D array.
    onset_velocities: Velocities of onsets scaled from [0, 1].
    active_velocities: Velocities of active notes scaled from [0, 1].
    offsets: An offset-only pianoroll as a 2D array.
    control_changes: Control change onsets as a 2D array (time, control number)
      with 0 when there is no onset and (control_value + 1) when there is.
  """
  roll = np.zeros((int(sequence.total_time * frames_per_second + 1),
                   max_pitch - min_pitch + 1),
                  dtype=np.float32)

  roll_weights = np.ones_like(roll)

  onsets = np.zeros_like(roll)
  offsets = np.zeros_like(roll)

  control_changes = np.zeros(
      (int(sequence.total_time * frames_per_second + 1), 128), dtype=np.int32)

  def frames_from_times(start_time, end_time):
    """Converts start/end times to start/end frames."""
    # Will round down because note may start or end in the middle of the frame.
    start_frame = int(start_time * frames_per_second)
    start_frame_occupancy = (start_frame + 1 - start_time * frames_per_second)
    # check for > 0.0 to avoid possible numerical issues
    if (min_frame_occupancy_for_label > 0.0 and
        start_frame_occupancy < min_frame_occupancy_for_label):
      start_frame += 1

    end_frame = int(math.ceil(end_time * frames_per_second))
    end_frame_occupancy = end_time * frames_per_second - start_frame - 1
    if (min_frame_occupancy_for_label > 0.0 and
        end_frame_occupancy < min_frame_occupancy_for_label):
      end_frame -= 1
      # can be a problem for very short notes
      end_frame = max(start_frame, end_frame)

    return start_frame, end_frame

  velocities_roll = np.zeros_like(roll, dtype=np.float32)

  for note in sorted(sequence.notes, key=lambda n: n.start_time):
    if note.pitch < min_pitch or note.pitch > max_pitch:
      tf.logging.warn('Skipping out of range pitch: %d', note.pitch)
      continue
    start_frame, end_frame = frames_from_times(note.start_time, note.end_time)

    # label onset events. Use a window size of onset_window to account of
    # rounding issue in the start_frame computation.
    onset_start_time = note.start_time + onset_delay_ms / 1000.
    onset_end_time = note.end_time + onset_delay_ms / 1000.
    if onset_mode == 'window':
      onset_start_frame_without_window, _ = frames_from_times(
          onset_start_time, onset_end_time)

      onset_start_frame = max(0,
                              onset_start_frame_without_window - onset_window)
      onset_end_frame = min(onsets.shape[0],
                            onset_start_frame_without_window + onset_window + 1)
    elif onset_mode == 'length_ms':
      onset_end_time = min(onset_end_time,
                           onset_start_time + onset_length_ms / 1000.)
      onset_start_frame, onset_end_frame = frames_from_times(
          onset_start_time, onset_end_time)
    else:
      raise ValueError('Unknown onset mode: {}'.format(onset_mode))

    # label offset events.
    offset_start_time = min(note.end_time,
                            sequence.total_time - offset_length_ms / 1000.)
    offset_end_time = offset_start_time + offset_length_ms / 1000.
    offset_start_frame, offset_end_frame = frames_from_times(
        offset_start_time, offset_end_time)
    offset_end_frame = max(offset_end_frame, offset_start_frame + 1)

    if not onset_overlap:
      start_frame = onset_end_frame
      end_frame = max(start_frame + 1, end_frame)

    offsets[offset_start_frame:offset_end_frame, note.pitch - min_pitch] = 1.0
    onsets[onset_start_frame:onset_end_frame, note.pitch - min_pitch] = 1.0
    roll[start_frame:end_frame, note.pitch - min_pitch] = 1.0

    if note.velocity > max_velocity:
      raise ValueError('Note velocity exceeds max velocity: %d > %d' %
                       (note.velocity, max_velocity))

    velocities_roll[start_frame:end_frame, note.pitch -
                    min_pitch] = float(note.velocity) / max_velocity
    roll_weights[onset_start_frame:onset_end_frame, note.pitch - min_pitch] = (
        onset_upweight)
    roll_weights[onset_end_frame:end_frame, note.pitch - min_pitch] = [
        onset_upweight / x for x in range(1, end_frame - onset_end_frame + 1)
    ]

    if add_blank_frame_before_onset:
      if start_frame > 0:
        roll[start_frame - 1, note.pitch - min_pitch] = 0.0
        roll_weights[start_frame - 1, note.pitch - min_pitch] = 1.0

  for cc in sequence.control_changes:
    frame, _ = frames_from_times(cc.time, 0)
    if frame < len(control_changes):
      control_changes[frame, cc.control_number] = cc.control_value + 1

  return Pianoroll(
      active=roll,
      weights=roll_weights,
      onsets=onsets,
      onset_velocities=velocities_roll * onsets,
      active_velocities=velocities_roll,
      offsets=offsets,
      control_changes=control_changes)

