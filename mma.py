#! /usr/bin/env python
# -*- coding: ascii -*-

# Copyright (c) 2014 Franco Bugnano

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Based on Samplicity v0.4
# Samplicity Copyright (C) 2012 Andrii Magalich

from __future__ import (division, print_function)

import sys
import os

import struct
import wave
import sndhdr
import timeit
import math
import argparse
import pprint


__version__ = '0.0.1'

VERSION = ''.join(['MMA v', __version__])

# MIDI notes go from 0 (C-1) to 127 (G9)
NOTES = []
scale = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
for octave in range(-1, 10):
	NOTES.extend([''.join([note, str(octave)]) for note in scale])
del scale
del octave
del note

# The first of the 96 notes (C0)
XI_FIRST_NOTE = NOTES.index('c0')

# The relative note with value 0 (C4)
XI_REL_NOTE_ZERO = NOTES.index('c4')

# Default values for SFZ parameters, as specified in the document (as a numeric value)
DEFAULT_LOKEY = 0
DEFAULT_HIKEY = 127
DEFAULT_PITCH_KEYCENTER = 60

SAMPLE_TYPE_8BIT = 0x00
SAMPLE_TYPE_16BIT = 0x10
SAMPLE_TYPE_32BIT = 0x20	# Non-standard (Samplicity v0.4 uses 0x10, but how is the tracker supposed to know whether it's 16 or 32 bit?)
SAMPLE_TYPE_MONO = 0x00
SAMPLE_TYPE_STEREO = 0x40	# Non-standard
SAMPLE_TYPE_NO_LOOP = 0x00
SAMPLE_TYPE_FWD_LOOP = 0x01
SAMPLE_TYPE_BIDI_LOOP = 0x02

XI_VERSION_FT2 = 0x0102
XI_VERSION_PSYTEXX = 0x0000	# Non-standard (In theory it is 0x5050, but Samplicity v0.4 uses 0x0000)


def pad_name(name, length, pad=' ', dir='right'):
	if dir == 'right':
		return ''.join([name, pad * length])[:length]
	else:
		return ''.join([name, pad * length])[-length:]


def identify_sample(sample_path, options):
	what = sndhdr.what(sample_path)
	if not what:
		print('This is not wav file:', sample_path)
		sys.exit(1)

	(format_, sampling_rate, channels, frames_count, bits_per_sample) = what

	if format_ != 'wav':
		print('This is not wav file:', sample_path)
		sys.exit(1)

	sample = wave.open(sample_path, 'rb')

	# The values returned by the wave module are more reliable than those returned by the sndhdr module
	sampling_rate = sample.getframerate()
	channels = sample.getnchannels()
	frames_count = sample.getnframes()
	byte = sample.getsampwidth()
	bits_per_sample = byte * 8

	sample_type = 0

	if sampling_rate == 0:
		print('Unknown sampling rate for wav file:', sample_path)
		sys.exit(1)

	if frames_count == -1:
		print('Unknown frame count for wav file:', sample_path)
		sys.exit(1)

	xi_version = XI_VERSION_FT2

	if channels == 1:
		text_type = 'mono'
		sample_type |= SAMPLE_TYPE_MONO
	elif channels == 2:
		text_type = 'stereo'
		if options.enable_stereo:
			sample_type |= SAMPLE_TYPE_STEREO
			xi_version = XI_VERSION_PSYTEXX
		else:
			print('/' * 80)
			print('stereo samples are not supported')
			print('Use the --enable-stereo option to enable non-standard stereo samples')
			print('/' * 80)
			sys.exit(1)
	else:
		text_type = '{0}-channels'.format(channels)
		print('/' * 80)
		print('{0}-channels samples are not supported'.format(channels))
		print('/' * 80)
		sys.exit(1)

	# Check the bit rate last, as bit-rate conversion should be the last operation to apply on the samples
	if bits_per_sample == 8:
		sample_type |= SAMPLE_TYPE_8BIT
	elif bits_per_sample == 16:
		sample_type |= SAMPLE_TYPE_16BIT
	else:
		if (bits_per_sample == 32) and (options.enable_32_bit):
			sample_type |= SAMPLE_TYPE_32BIT
			xi_version = XI_VERSION_PSYTEXX
		else:
			print('/' * 80)
			print('{0}-bit samples are not supported'.format(bits_per_sample))
			if bits_per_sample == 32:
				print('Use the --enable-32-bit option to enable non-standard 32-bit samples')
			print('/' * 80)
			sys.exit(1)

	print('*', bits_per_sample, 'bit', text_type, 'sample "', sample_path, '"', int((byte * frames_count * channels) / 1024), 'kB')

	return {
		'sampling_rate': sampling_rate,
		'sample_bittype': byte,
		'channels': channels,
		'sample_length': frames_count,
		'sample_type': sample_type,
		'xi_version': xi_version,
	}


def write_delta_sample(sample_path, fp):
	sample = wave.open(sample_path, 'rb')

	byte = sample.getsampwidth()
	if byte == 1:
		bittype = 'B'
		scissors = 0xFF
	elif byte == 2:
		bittype = 'H'
		scissors = 0xFFFF
	elif byte == 4:
		bittype = 'I'
		scissors = 0xFFFFFFFF
	else:
		print('/' * 80)
		print('{0}-bit samples are not supported'.format(byte * 8))
		print('/' * 80)
		sys.exit(1)

	frames = []
	while True:
		r = sample.readframes(512)
		if not r:
			break

		frames.extend(struct.unpack(''.join(['<{0}'.format(int(len(r) / byte)), bittype]), r))

	sample.close()

	delta = 0
	for frame in frames:
		original = frame
		frame = (frame - delta) & scissors
		delta = original
		fp.write(struct.pack(''.join(['<', bittype]), frame))


# FixedMUL, getc4spd and convertc4spd are magic functions taken from the MilkyTracker source code
mp_sbyte = lambda x: struct.unpack('<b', struct.pack('<B', (int(x) & 0xFF)))[0]

def FixedMUL(a, b):
	return ((a * b) >> 16)

def getc4spd(relnote, finetune):
	table = [
		65536,69432,73561,77935,82570,87480,92681,98193,104031,110217,116771,123715,
		65536,65565,65595,65624,65654,65684,65713,65743,65773,65802,65832,65862,65891,
		65921,65951,65981,66010,66040,66070,66100,66130,66160,66189,66219,66249,66279,
		66309,66339,66369,66399,66429,66459,66489,66519,66549,66579,66609,66639,66669,
		66699,66729,66759,66789,66820,66850,66880,66910,66940,66971,67001,67031,67061,
		67092,67122,67152,67182,67213,67243,67273,67304,67334,67365,67395,67425,67456,
		67486,67517,67547,67578,67608,67639,67669,67700,67730,67761,67792,67822,67853,
		67883,67914,67945,67975,68006,68037,68067,68098,68129,68160,68190,68221,68252,
		68283,68314,68344,68375,68406,68437,68468,68499,68530,68561,68592,68623,68654,
		68685,68716,68747,68778,68809,68840,68871,68902,68933,68964,68995,69026,69057,
		69089,69120,69151,69182,69213,69245,69276,69307,69339,69370,69401
	]

	c4spd = 8363
	xmfine = mp_sbyte(finetune)

	octave = mp_sbyte((relnote + 96) / 12)
	note = mp_sbyte((relnote + 96) % 12)

	o2 = mp_sbyte(octave - 8)

	if xmfine < 0:
		xmfine = mp_sbyte(xmfine + 128)
		note -= 1
		if note < 0:
			note += 12
			o2 -= 1

	if o2 >= 0:
		c4spd <<= o2
	else:
		c4spd >>= -o2

	f = FixedMUL(table[note], c4spd)
	return FixedMUL(f, table[xmfine+12])

def convertc4spd(c4spd):
	xmfine = 0
	cl = 0
	ch = 0
	ebp = 0xFFFFFFFF
	ebx = c4spd

	aloop = True
	while aloop:
		aloop = False
		c4s2 = ebx & 0xFFFFFFFF
		c4s = getc4spd(cl-48, 0) & 0xFFFFFFFF
		if c4s < c4s2:
			s = c4s2
			c4s2 = c4s
			c4s = s
		dc4 = (c4s - c4s2) & 0xFFFFFFFF
		if dc4 < ebp:
			ebp = dc4 & 0xFFFFFFFF
			ch = cl
			cl = mp_sbyte(cl + 1)
			if cl < 119:
				aloop = True
		if not aloop:
			cl = 0

	aloop2 = True
	while aloop2:
		aloop2 = False
		c4s2 = ebx & 0xFFFFFFFF
		c4s = getc4spd(ch-48, xmfine) & 0xFFFFFFFF
		if c4s < c4s2:
			s = c4s2
			c4s2 = c4s
			c4s = s
		dc4 = (c4s - c4s2) & 0xFFFFFFFF
		if dc4 < ebp:
			ebp = dc4 & 0xFFFFFFFF
			cl = mp_sbyte(xmfine)
		xmfine += 1
		if xmfine < 256:
			aloop2 = True

	ch = mp_sbyte(ch - 48)
	finetune = cl
	relnote = ch
	return (finetune, relnote)


# path_insensitive and _path_insensitive functions by Chris Morgan (originally on the PortableApps development toolkit)
def path_insensitive(path):
	"""
	Get a case-insensitive path for use on a case sensitive system.

	>>> path_insensitive('/Home')
	'/home'
	>>> path_insensitive('/Home/chris')
	'/home/chris'
	>>> path_insensitive('/HoME/CHris/')
	'/home/chris/'
	>>> path_insensitive('/home/CHRIS')
	'/home/chris'
	>>> path_insensitive('/Home/CHRIS/.gtk-bookmarks')
	'/home/chris/.gtk-bookmarks'
	>>> path_insensitive('/home/chris/.GTK-bookmarks')
	'/home/chris/.gtk-bookmarks'
	>>> path_insensitive('/HOME/Chris/.GTK-bookmarks')
	'/home/chris/.gtk-bookmarks'
	>>> path_insensitive("/HOME/Chris/I HOPE this doesn't exist")
	"/HOME/Chris/I HOPE this doesn't exist"
	"""

	return _path_insensitive(path) or path

def _path_insensitive(path):
	"""
	Recursive part of path_insensitive to do the work.
	"""

	if path == '' or os.path.exists(path):
		return path

	base = os.path.basename(path)  # may be a directory or a file
	dirname = os.path.dirname(path)

	suffix = ''
	if not base:  # dir ends with a slash?
		if len(dirname) < len(path):
			suffix = path[:len(path) - len(dirname)]

		base = os.path.basename(dirname)
		dirname = os.path.dirname(dirname)

	if not os.path.exists(dirname):
		dirname = _path_insensitive(dirname)
		if not dirname:
			return

	# at this point, the directory exists but not the file

	try:  # we are expecting dirname to be a directory, but it could be a file
		files = os.listdir(dirname)
	except OSError:
		return

	baselow = base.lower()
	try:
		basefinal = next(fl for fl in files if fl.lower() == baselow)
	except StopIteration:
		return

	if basefinal:
		return os.path.join(dirname, basefinal) + suffix
	else:
		return

def path_local(path):
	if os.sep == '\\':
		return path
	else:
		return path.replace('\\', os.sep)


# Function taken from http://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
def get_n_indices(m, n):
	return [(int((i * n) / m) + int(n / (2 * m))) for i in range(m)]


class SFZ_region(object):
	def __init__(self):
		self.sfz_params = {}
		self.wav_params = {}

	def validate(self):
		# A region without a sample is useless
		if 'sample' not in self.sfz_params:
			return False

		if 'tune' not in self.sfz_params:
			self.sfz_params['tune'] = 0

		# The key parameter sets the lokey, hikey, and pitch_keycenter to the same value
		if 'key' in self.sfz_params:
			self.sfz_params['lokey'] = self.sfz_params['key']
			self.sfz_params['hikey'] = self.sfz_params['key']
			self.sfz_params['pitch_keycenter'] = self.sfz_params['key']

		# Convert note names to numeric values
		for param in ('lokey', 'hikey', 'pitch_keycenter'):
			if param in self.sfz_params:
				value = self.sfz_params[param]
				try:
					if self.sfz_params[param].isdigit():
						self.sfz_params[param] = int(value)
					else:
						self.sfz_params[param] = NOTES.index(value.lower())

					if (self.sfz_params[param] < 0) or (self.sfz_params[param] >= len(NOTES)):
						raise ValueError
				except ValueError:
					print('ERROR: Invalid {} value for sample {}: {}'.format(param, self.sfz_params['sample'], value))
					sys.exit(1)

		# lokey, hikey, and pitch_keycenter have a default value if not set
		if 'lokey' not in self.sfz_params:
			self.sfz_params['lokey'] = DEFAULT_LOKEY

		if 'hikey' not in self.sfz_params:
			self.sfz_params['hikey'] = DEFAULT_HIKEY

		if 'pitch_keycenter' not in self.sfz_params:
			self.sfz_params['pitch_keycenter'] = DEFAULT_PITCH_KEYCENTER

		if self.sfz_params['lokey'] > self.sfz_params['hikey']:
			print('Notice: swapping lokey and hikey for region:', self.sfz_params['sample'])
			self.sfz_params['hikey'], self.sfz_params['lokey'] = (self.sfz_params['lokey'], self.sfz_params['hikey'])

		return True

	def load_audio(self, cwd, options):
		# Strip the leading path separators for badly formed sfz files (the sfz standard mandates relative paths)
		sample_path = path_local(self.sfz_params['sample']).lstrip(os.sep)
		self.wav_params['sample_path'] = path_insensitive(os.path.normpath(os.path.join(cwd, sample_path)))
		self.wav_params.update(identify_sample(self.wav_params['sample_path'], options))


def parse_sfz(filename, options):
	regions = []

	group_params = {}
	last_chunk = None
	curr_region = None
	in_region = False
	in_group = False

	lineno = 0
	fp = open(filename, 'rU')
	for line in fp:
		lineno += 1

		# remove comments
		comment_pos = line.find('//')
		if comment_pos >= 0:
			line = line[:comment_pos]

		line = line.strip()

		# blank line - nothing to do here
		if not line:
			continue

		# now split line in chunks by spaces
		chunks = line.split(' ')
		for chunk in chunks:
			# As the sfz specs state, chunks beginning with the character '/' are comments,
			# so let's not process the rest of the line.
			# Notice that I look for the '/' character only as the first character of the chunk,
			# in order to be compatible with the '/' character as a path separator.
			# I don't know if the '/' path separator is supported in the sfz standard or not.
			if chunk.startswith('/'):
				break

			if chunk == '<group>':
				# it's a group - lets remember the following
				last_chunk = None
				in_group = True
				in_region = False
				group_params = {}
			elif chunk == '<region>':
				# it's a region - save the following and add group data
				curr_region = SFZ_region()
				regions.append(curr_region)
				if in_group:
					curr_region.sfz_params.update(group_params)

				last_chunk = None
				in_region = True
			else:
				# this should be the assignment
				segments = chunk.split('=', 1)
				if len(segments) != 2:
					# maybe, we can just append this data to the previous chunk
					if last_chunk is not None:
						curr_region.sfz_params[last_chunk[0]] = ' '.join([curr_region.sfz_params[last_chunk[0]], segments[0]])
						segments = (last_chunk[0], curr_region.sfz_params[last_chunk[0]])
					else:
						print('Ambiguous spaces in SFZ file:', filename, 'at line:', lineno)
						sys.exit(1)

				if in_region:
					curr_region.sfz_params[segments[0]] = segments[1]
				elif in_group:
					group_params[segments[0]] = segments[1]

				last_chunk = segments

	fp.close()

	cwd = os.path.dirname(filename)

	# complete samples info
	delete_regions = []
	for (i, region) in enumerate(regions):
		if region.validate():
			region.load_audio(cwd, options)
		else:
			# Invalid region, mark it for deletion (last index first)
			delete_regions.insert(0, i)

	# Delete invalid regions
	if delete_regions:
		print('/' * 80)
		print('Notice: some regions are invalid and ignored')
		print('/' * 80)

		for i in delete_regions:
			del regions[i]

	return regions


def magic(filename, xi_filename, options):
	start = timeit.default_timer()

	head, tail = os.path.split(filename)
	root, ext = os.path.splitext(tail)

	regions = parse_sfz(filename, options)

	# The regions are sorted by lokey, and by the inverse of hikey, in order to have the regions
	# that span the wider range of notes first
	regions.sort(key=lambda x: [x.sfz_params['lokey'], -(x.sfz_params['hikey'])])

	# Delete fully overlapping regions
	last_region = None
	delete_regions = []
	for (i, region) in enumerate(regions):
		if last_region:
			if (region.sfz_params['lokey'] >= last_region.sfz_params['lokey']) and (region.sfz_params['hikey'] <= last_region.sfz_params['hikey']):
				# Overlapping region, mark it for deletion (last index first)
				delete_regions.insert(0, i)
			elif region.sfz_params['lokey'] > last_region.sfz_params['lokey']:
				# lokey changed, mark this as the wider region
				last_region = region
		else:
			last_region = region

	if delete_regions:
		overlapping = []
		for i in delete_regions:
			overlapping.insert(0, regions[i].sfz_params['sample'])
			del regions[i]

		print('/' * 80)
		print('Notice: some regions are fully overlapping and would be overwritten')
		print('Skipping:')
		pprint.pprint(overlapping)
		print('/' * 80)

	if not regions:
		print('No regions found in file:', tail)
		return

	if len(regions) > options.max_samples:
		# Keep only the maximum allowed number of regions, but we are keeping them evenly spaced, in order
		# to support the widest possible range of notes
		if options.drumset:
			# For drumsets we keep only the first samples
			keep_regions = list(range(options.max_samples))
		else:
			# For normal instruments we evenly distribute samples
			keep_regions = get_n_indices(options.max_samples, len(regions))

		exclude_regions = [regions[i].sfz_params['sample'] for i in range(len(regions)) if i not in keep_regions]
		regions = [regions[i] for i in keep_regions]

		print('/' * 80)
		print('Too many samples in file:', tail, '(no more than {0} samples supported)'.format(options.max_samples))
		print('Skipping:')
		pprint.pprint(exclude_regions)
		print('/' * 80)

	notes_samples = [0] * 96

	if not options.drumset:
		# Extend the first and last regions in order to have all the notes covered
		if regions[0].sfz_params['lokey'] > XI_FIRST_NOTE:
			print('Notice: the first region has been extended from {} to {}'.format(NOTES[regions[0].sfz_params['lokey']], NOTES[XI_FIRST_NOTE]))
			regions[0].sfz_params['lokey'] = XI_FIRST_NOTE

		if regions[-1].sfz_params['hikey'] < (XI_FIRST_NOTE + (len(notes_samples) - 1)):
			print('Notice: the last region has been extended from {} to {}'.format(NOTES[regions[-1].sfz_params['hikey']], NOTES[XI_FIRST_NOTE + (len(notes_samples) - 1)]))
			regions[-1].sfz_params['hikey'] = XI_FIRST_NOTE + (len(notes_samples) - 1)

	# Fill the gaps, and resolve the overlapping notes,
	# and while we're here, find the right XI version number
	xi_version = XI_VERSION_FT2
	last_region = None
	for region in regions:
		adjust = False
		if last_region:
			if region.sfz_params['lokey'] <= last_region.sfz_params['hikey']:
				# Overlapping regions
				print('Notice: overlapping notes from {} to {}'.format(NOTES[region.sfz_params['lokey']], NOTES[last_region.sfz_params['hikey']]))
				adjust = True
			elif region.sfz_params['lokey'] > (last_region.sfz_params['hikey'] + 1):
				# Gap between regions
				print('Notice: gap between {} and {}'.format(NOTES[last_region.sfz_params['hikey']], NOTES[region.sfz_params['lokey']]))
				adjust = True

			# Both gaps and overlapping regions are fixed by extending them to a middle point
			if adjust and not options.drumset:
				midpoint = int((region.sfz_params['lokey'] + last_region.sfz_params['hikey']) / 2)
				print('Adjusting to {}'.format(NOTES[midpoint]))
				last_region.sfz_params['hikey'] = midpoint
				region.sfz_params['lokey'] = midpoint + 1

		if region.wav_params['xi_version'] != XI_VERSION_FT2:
			xi_version = region.wav_params['xi_version']

		last_region = region

	# Map the samples to the corresponding notes
	overlapping = set()
	ignored = set()
	for (i, region) in enumerate(regions):
		lo = region.sfz_params['lokey']
		hi = region.sfz_params['hikey']
		for note in range(lo, hi + 1):
			xi_note = note - XI_FIRST_NOTE
			if (xi_note >= 0) and (xi_note < len(notes_samples)):
				if notes_samples[xi_note]:
					overlapping.add(note)
				else:
					notes_samples[xi_note] = i
			else:
				ignored.add(note)

	if overlapping:
		print('/' * 80)
		print('Notice: some regions are overlapping and would be overwritten')
		pprint.pprint([NOTES[x] for x in sorted(overlapping)])
		print('/' * 80)

	if ignored:
		print('/' * 80)
		print('Notice: some notes are out of range and ignored')
		pprint.pprint([NOTES[x] for x in sorted(ignored)])
		print('/' * 80)

	# create xi file
	temp_filename = ''.join([xi_filename, '.temp'])
	fp = open(temp_filename, 'wb')

	# -------------------------------------------------------------- file header
	fp.write(struct.pack('<21s22sb20sh',
		'Extended Instrument: ',
		pad_name(root, 22),
		0x1A,
		pad_name(VERSION, 20),
		xi_version
	))

	# -------------------------------------------------------------- inst header

	# ADSR envelope:
	# Attack time: time to go from the minimum to the maximum value (2 points: 0 time, minimum value -- attack time, maximum value)
	# Decay time: time to go from the maximum value to the sustain value (1 point: decay time, sustain value)
	# Sustain level: Level at which the note is played while ON (no points, but it's the sustain point parameter)
	# Release time: time to go from the sustain value to the minimum value (1 point: release time, release value)
	# The sfz standard also has a Delay time before the Attack, and a Hold time between Attack and Decay.

	# seconds-to-ticks converter
	# Why 50?
	stt = 50

	# volume envelope
	volume_ticks = 0
	volume_level = 0
	volume_envelope = []
	vol_sustain_point = None

	# Use the first region to generate the envelope
	region = regions[0]

	if 'ampeg_start' in region.sfz_params:
		volume_level = int((float(region.sfz_params['ampeg_start']) * 0x40) / 100)

	if 'ampeg_delay' in region.sfz_params:
		volume_envelope.append(volume_ticks)
		volume_envelope.append(volume_level)

		volume_ticks += int(float(region.sfz_params['ampeg_delay']) * stt)

	if 'ampeg_attack' in region.sfz_params:
		volume_envelope.append(volume_ticks)
		volume_envelope.append(volume_level)

		volume_ticks += int(float(region.sfz_params['ampeg_attack']) * stt)

	# After the attack time, the volume level is at its maximum value
	volume_level = 0x40

	if 'ampeg_hold' in region.sfz_params:
		volume_envelope.append(volume_ticks)
		volume_envelope.append(volume_level)

		volume_ticks += int(float(region.sfz_params['ampeg_hold']) * stt)

	if 'ampeg_decay' in region.sfz_params:
		volume_envelope.append(volume_ticks)
		volume_envelope.append(volume_level)

		volume_ticks += int(float(region.sfz_params['ampeg_decay']) * stt)

	# After the decay time, the volume level is at the sustain level
	if 'ampeg_sustain' in region.sfz_params:
		volume_level = int((float(region.sfz_params['ampeg_sustain']) * 0x40) / 100)

	if volume_envelope:
		volume_envelope.append(volume_ticks)
		volume_envelope.append(volume_level)
		vol_sustain_point = int(len(volume_envelope) / 2) - 1

	# After the sustain, the volume level is at its minimum value
	volume_level = 0

	if 'ampeg_release' in region.sfz_params:
		volume_ticks += int(float(region.sfz_params['ampeg_release']) * stt)
		if volume_envelope:
			volume_envelope.append(volume_ticks)
			volume_envelope.append(volume_level)
		else:
			# If the envelope has not been created yet, set the first point for the sustain
			volume_envelope.append(0)
			volume_envelope.append(0x40)
			vol_sustain_point = int(len(volume_envelope) / 2) - 1

			# and then set the release point
			volume_envelope.append(volume_ticks)
			volume_envelope.append(volume_level)

	if volume_ticks > 512:
		for i in range(int(len(volume_envelope) / 2)):
			volume_envelope[2 * i] = int((volume_envelope[2 * i] * 512) / volume_ticks)

		print('/' * 80)
		print('Too long envelope:', volume_ticks, 'ticks, shrinked to 512')
		print('/' * 80)

	# Sample number for notes 1..96
	fp.write(struct.pack('<{}b'.format(len(notes_samples)), *(notes_samples)))

	# 12 volume envelope points
	fp.write(struct.pack('<{}h'.format(len(volume_envelope)), *(volume_envelope)))
	fp.write(struct.pack('<{}h'.format(24 - len(volume_envelope)), *([0] * (24 - len(volume_envelope)))))

	# 12 panning envelope points
	fp.write(struct.pack('<24h', *([0] * 24)))

	# Number of volume points
	fp.write(struct.pack('<b', int(len(volume_envelope) / 2)))

	# Number of panning points
	fp.write(struct.pack('<b', 0))

	# Volume sustain point
	if vol_sustain_point is not None:
		fp.write(struct.pack('<b', vol_sustain_point))
	else:
		fp.write(struct.pack('<b', 0))

	# Volume loop start point
	fp.write(struct.pack('<b', 0))

	# Volume loop end point
	fp.write(struct.pack('<b', 0))

	# Panning sustain point
	fp.write(struct.pack('<b', 0))

	# Panning loop start point
	fp.write(struct.pack('<b', 0))

	# Panning loop end point
	fp.write(struct.pack('<b', 0))

	# Volume type;   b0=on, b1=sustain, b2=loop
	if volume_envelope:
		volume_type = 0x03
	else:
		volume_type = 0

	fp.write(struct.pack('<b', volume_type))

	# Panning type;  b0=on, b1=sustain, b2=loop
	fp.write(struct.pack('<b', 0))

	# Vibrato type
	fp.write(struct.pack('<b', 0))

	# Vibrato sweep
	fp.write(struct.pack('<b', 0))

	# Vibrato depth
	fp.write(struct.pack('<b', 0))

	# Vibrato rate
	fp.write(struct.pack('<b', 0))

	# Volume fadeout (0..fff)
	fp.write(struct.pack('<h', 0))

	# ????? (Zeroes or extened info for PsyTexx (vol,finetune,pan,relative,flags))
	fp.write(struct.pack('<22b', *([0] * 22)))

	# Number of Samples
	fp.write(struct.pack('<h', len(regions)))

	# ---------------------------------------------------------- sample headers
	for region in regions:
		fp.write(struct.pack('<i', region.wav_params['sample_length'] * region.wav_params['sample_bittype'] * region.wav_params['channels']))  # sample length
		fp.write(struct.pack('<2i', 0, 0))  # sample loop start and end
		# volume
		if 'volume' in region.sfz_params:
			fp.write(struct.pack('<B', math.floor(255 * math.exp(float(region.sfz_params['volume']) / 10) / math.exp(0.6))))	# 'cause volume is in dB
		else:
			fp.write(struct.pack('<B', 255))

		# Get the relative note and finetune based on the sampling rate of the sample
		finetune, relnote = convertc4spd(region.wav_params['sampling_rate'])
		relnote += XI_REL_NOTE_ZERO

		fp.write(struct.pack('<b', finetune + int(region.sfz_params['tune'])))  # finetune (signed!)

		fp.write(struct.pack('<b', region.wav_params['sample_type']))  # sample type

		#panning (unsigned!)
		if 'pan' in region.sfz_params:
			fp.write(struct.pack('<B', (float(region.sfz_params['pan']) + 100) * 255 / 200))
		else:
			fp.write(struct.pack('<B', 128))

		# relative note - transpose c4 ~ 00
		fp.write(struct.pack('<b', relnote - region.sfz_params['pitch_keycenter']))

		# Sample Name, padded w/ zeroes
		root, ext = os.path.splitext(os.path.basename(path_local(region.sfz_params['sample'])))
		sample_name = pad_name(root, 22, '\0')

		fp.write(struct.pack('<b', len(sample_name.rstrip('\0'))))
		fp.write(struct.pack('<22s', sample_name))

	# ------------------------------------------------------------- sample data
	for region in regions:
		write_delta_sample(region.wav_params['sample_path'], fp)

	print(len(regions), 'samples')
	print(int(fp.tell() / 1024), 'kB written in file "', os.path.basename(xi_filename), '" during', timeit.default_timer() - start, 'seconds')

	fp.close()

	if os.path.exists(xi_filename):
		os.remove(xi_filename)

	os.rename(temp_filename, xi_filename)


def main(argv):
	parser = argparse.ArgumentParser(prog=argv[0], description='Convert .sfz samples to .xi format')
	parser.add_argument('--version', action='version', version=VERSION)
	parser.add_argument('-f', '--force', help='force reconversion', action='store_true')
	parser.add_argument('-d', '--output-dir', help='set output directory')
	parser.add_argument('-m', '--max-samples', help='set maximum number of samples (default: 16)', type=int, default=16)
	parser.add_argument('-s', '--enable-stereo', help='enable support for stereo samples', action='store_true')
	parser.add_argument('-4', '--enable-32-bit', help='enable support for 32-bit samples', action='store_true')
	parser.add_argument('-r', '--drumset', help='the specified sfz files are drumsets', action='store_true')
	parser.add_argument('sfz_file', help='sfz file(s) to convert', nargs='+')
	options = parser.parse_args(argv[1:])

	if not options.output_dir:
		options.output_dir = os.getcwd()
	else:
		options.output_dir = os.path.normpath(options.output_dir)

	if not os.path.isdir(options.output_dir):
		print('ERROR: Invalid output directory')
		return 2

	# As there are only 96 notes in the instrument header and at most 1 sample per note,
	# it doesn't make any sense to support more than 96 samples per instrument
	if (options.max_samples < 1) or (options.max_samples > 96):
		print('ERROR: Invalid maximum number of samples (valid from 1 to 96)')
		return 2

	start_time = timeit.default_timer()
	converted = 0
	for arg in options.sfz_file:
		head, tail = os.path.split(arg)
		root, ext = os.path.splitext(tail)
		xi_filename = os.path.join(options.output_dir, ''.join([root, '.xi']))
		if (not os.path.exists(xi_filename)) or options.force:
			print('-' * 80)
			print('Converting "', tail, '"')
			print('-' * 80)
			magic(arg, xi_filename, options)
			converted += 1
		else:
			print('File', tail, 'is already converted!')

	print('')
	print(converted, 'files converted in', timeit.default_timer() - start_time, 'seconds')


if __name__ == '__main__':
	sys.exit(main(sys.argv))

