from operator import itemgetter
import numpy as np
import midi
import os

''' 
	Used to convert musical sequence data into MIDI. 
	The data must follow the encoding scheme detailed
	in my undergraduate thesis.
'''

MAX_MIDI_PITCH = 128
MIDI_VELOCITY = 75

def decode_symbol(symbol):
	symbol -= 3
	midi_pitch = symbol//2
	note_type = symbol % 2 + 1
	return midi_pitch, note_type

def get_time_steps(note_sequence):
	print('Retrieving time steps from note sequence...')
	time_steps = {}
	time_step = []
	found_first_note = False
	step = 0
	
	for symbol in note_sequence:
		
		if symbol in [0,1]:
			continue
		
		elif symbol != 2: 
			found_first_note = True
			time_step += [symbol]
		
		elif symbol == 2 and found_first_note: 
			time_steps[step] = time_step
			step += 1
			time_step = []
		
	time_steps[step] = time_step
	
	return time_steps

def build_note_matrix(time_steps):
	print('Building (%d x %d) note matrix from time steps...' % (MAX_MIDI_PITCH, len(time_steps)))

	total_steps = len(time_steps)
	note_matrix = np.zeros((MAX_MIDI_PITCH, total_steps))
	
	sorted_time_steps = sorted(time_steps.items(), key = itemgetter(0))
	for step_num, time_step in sorted_time_steps:
		for symbol in time_step:
			
			midi_pitch, note_type = decode_symbol(symbol)
			note_matrix[midi_pitch][step_num] = note_type
			
	return note_matrix
		  
	
def trim_note_matrix(note_matrix):
	first_step = 0
	while not np.any(note_matrix[:,first_step] > 0):
		first_step += 1
		
	last_step = note_matrix.shape[1] - 1
	while not np.any(note_matrix[:,last_step] > 0):
		last_step -= 1
		
	return note_matrix[:, first_step:last_step + 1]

def build_midi_pattern(note_matrix, midi_ticks_per_step = 50):
	print('Building MIDI pattern from note matrix...')
	note_matrix = trim_note_matrix(note_matrix)
	
	pattern = midi.Pattern()
	track = midi.Track()
	pattern.append(track)
	
	tick_distance = 0 
	previous_time_step = np.zeros(MAX_MIDI_PITCH) 
	
	for cur_time_step in note_matrix.T:
		for midi_pitch in range(MAX_MIDI_PITCH):
			note_type = cur_time_step[midi_pitch]
			
			# no track event
			if note_type == 2 or (note_type == 0 and previous_time_step[midi_pitch] == 0):
				continue
			
			elif note_type == 0: 
				off = midi.NoteOffEvent(tick = tick_distance+2, pitch = midi_pitch)
				track.append(off)
				tick_distance = 0
				continue
				
			elif previous_time_step[midi_pitch] != 0:
				off = midi.NoteOffEvent(tick = tick_distance, pitch = midi_pitch)
				track.append(off)
				tick_distance = 2

			on = midi.NoteOnEvent(tick = tick_distance,
					      velocity = MIDI_VELOCITY,
					      pitch = midi_pitch)
			track.append(on)
			tick_distance = 0 
		
		previous_time_step = cur_time_step
		tick_distance += midi_ticks_per_step
	
	for midi_pitch in range(MAX_MIDI_PITCH):
		note_type = previous_time_step[midi_pitch]
		if note_type != 0:
			off = midi.NoteOffEvent(tick = tick_distance, pitch = midi_pitch)
			track.append(off)
			tick_distance = 0
	
	eot = midi.EndOfTrackEvent(tick = 1)
	track.append(eot)
	print('MIDI pattern completed.')
	return pattern

def sequence_to_midi(sequence, dir_path, file_name = 'sample'):
	print('Converting sequence to MIDI...')
	timeSteps = get_time_steps(sequence)
	note_matrix = build_note_matrix(timeSteps)
	midi_pattern = build_midi_pattern(note_matrix)
	filepath = os.path.join(dir_path, '%s.mid' % file_name) 
	midi.write_midifile(filepath, midi_pattern)
	print('Sequence to MIDI conversion successful.')

