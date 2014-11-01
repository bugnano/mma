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
import tempfile
import wave
import sndhdr
import timeit
import math
import shutil

from array import array


__version__ = '0.0.1'

VERSION = ''.join(['MMA v', __version__])

NOTES = []
scale = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
for i in range(1, 11):
	NOTES.extend([(note + str(i)) for note in scale])
del scale
del i
del note


def pad_name(name, length, pad=' ', dir='right'):
	if dir == 'right':
		return (name + pad * length)[:length]
	else:
		return (name + pad * length)[-length:]


def wrap(text, width):
	'''
	A word-wrap function that preserves existing line breaks
	and most spaces in the text. Expects that existing line
	breaks are posix newlines (\n).
	'''
	return reduce(lambda line, word, width=width: '%s%s%s' % (line, ' \n'[(len(line) - line.rfind('\n') - 1 + len(word.split('\n', 1)[0]) >= width)], word), text.split(' '))


class SFZ_region(dict):
	def read_wav(self, sample_path, cwd):
		(format_, sampling_rate, channels, frames_count, bits_per_sample) = sndhdr.what(cwd + sample_path)
		sample = wave.open(cwd + sample_path)

		if frames_count == -1:
			frames_count = sample.getnframes()

		if format_ != 'wav':
			print('This is not wav file:', sample_path)

		if channels == 1:
			text_type = 'mono'
			sample_type = 0b00010000
		elif channels == 2:
			text_type = 'stereo'
			sample_type = 0b01010000
		else:
			text_type = '{0}-channels'.format(channels)

		byte = int(bits_per_sample / 8)	# sample.getsampwidth()

		if byte == 3:  # for some reason
			print('*', (byte * 8), 'bit', text_type, 'sample "', sample_path, '"', int(byte * frames_count / (2 ** 9)), 'kB')
		else:
			print('*', (byte * 8), 'bit', text_type, 'sample "', sample_path, '"', int(byte * frames_count / (2 ** 10)), 'kB')

		if byte == 1:
			bittype = 'B'
			scissors = 0xFF
		elif byte == 2:
			bittype = 'H'
			scissors = 0xFFFF
		elif byte == 3:
			scissors = 0xFFFFFF
			bittype = 'I'
			print('/' * 80)
			print('24bit samples are not supported')
			print('/' * 80)
			# return ([], sample_type)
		elif byte == 4:
			scissors = 0xFFFFFFFF
			bittype = 'I'

		delta = 0
		frames = []
		total_len = byte * frames_count

		if byte == 3:  # need to treat this independently for some reason. maybe, python.wave bug?
			frames = struct.unpack('<{0}B'.format(total_len * 2), sample.readframes(total_len))
			# frames = []
			# for i in range(int(total_len / 2)):
			#	  bytes = struct.unpack('<6B', sample.readframes(1))
			#	  for j in range(int(len(bytes) / 3)):
			#		  frames.append(bytes[j] + bytes[j + 1] << 0xFF + bytes[j + 2] << 0xFFFF)
					# 'cause little-endian
			# bytes = struct.unpack('<{0}'.format(int(total_len / 2)) + bittype, sample.readframes(total_len))
			# for i in range(int(total_len / 3)):
			#	  frames.append(bytes[i] + bytes[i + 1] << 0xFF + bytes[i + 2] << 0xFFFF)
				# 'cause little-endian
		else:
			for i in range(int(total_len / (2 ** 9) + 1)):
				r = sample.readframes(2 ** 9)
				frames[len(frames):] = struct.unpack('<{0}'.format(int(len(r) / byte)) + bittype, r)

		sample.close()
		del sample

		ret = array(bittype)
		for frame in frames:
			original = frame
			frame = (frame - delta) & scissors
			delta = original
			ret.append(frame)

		frames = []
		del frames
		return (ret, sample_type, byte)

	def validate(self):
		if 'tune' not in self:
			self['tune'] = 0
		if 'key' in self:
			self['pitch_keycenter'] = self['key']
			self['lokey'] = self['key']
			self['hikey'] = self['key']
		if 'pitch_keycenter' in self:
			if 'lokey' not in self:
				self['lokey'] = self['pitch_keycenter']
			if 'hikey' not in self:
				self['hikey'] = self['pitch_keycenter']

		for key in ('pitch_keycenter', 'lokey', 'hikey'):
			if key in self and self[key].isdigit():
				self[key] = NOTES[int(self[key])]

	def load_audio(self, cwd):
		self['sample_path'] = self['sample'].replace('\\', '/')
		if self['sample_path'][-4:] == '.wav':
			(self['sample_data'], self['sample_type'], self['sample_bittype']) = self.read_wav(self['sample_path'], cwd)


class SFZ_instrument:
	def __init__(self, filename, cwd, tempdir):
		self.open(filename)

		self.regions = []
		self.group = {}
		self.last_chunk = None
		self.curr = -1
		self.in_region = -1
		self.in_group = False

		line = self.read()
		while len(line) > 0:
			self.parse_line(line)
			line = self.read()

		# complete samples info
		for region in self.regions:
			region.validate()
			lo = NOTES.index(region['lokey'])
			hi = NOTES.index(region['hikey'])
			region['notes'] = range(lo, hi + 1)
			region.load_audio(cwd)
			region['delta_sample'] = tempdir + str(timeit.default_timer()) + '.dat'
			region['sample_length'] = len(region['sample_data'])
			df = open(region['delta_sample'], 'w')

			if region['sample_bittype'] == 1:
				df.write(struct.pack('<{0}B'.format(len(region['sample_data'])), *(region['sample_data'])))
			elif region['sample_bittype'] == 2:
				df.write(struct.pack('<{0}H'.format(len(region['sample_data'])), *(region['sample_data'])))
			elif region['sample_bittype'] == 3:
				for byte in region['sample_data']:
					df.write(struct.pack('<3B', byte & 0xFF0000 >> 0xFFFF, byte & 0xFF00 >> 0xFF, byte & 0xFF))
			elif region['sample_bittype'] == 4:
				df.write(struct.pack('<{0}I'.format(len(region['sample_data'])), *(region['sample_data'])))

			df.close()
			region['sample_data'] = ''
			del region['sample_data']

	def open(self, filename):
		self.filename = filename
		self.fp = open(filename, 'r')
		return self.fp

	def close(self):
		self.fp.close()

	def read(self):
		return self.fp.readline()

	def parse_line(self, line):
		line = line.strip(' \r\n')
		comment_pos = line.find('//')  # remove comments
		if comment_pos >= 0:
			line = line[:comment_pos]
		if len(line) == 0:
			return	# blank line - nothing to do here
		# now split line in chunks by spaces
		chunks = line.split(' ')
		for chunk in chunks:
			if len(chunk) > 0:
				self.parse_chunk(chunk)

	def parse_chunk(self, chunk):
		if chunk == '<group>':	# it's a group - lets remember the following
			self.in_group = True
			self.in_region = False
			self.group = {}
		elif chunk == '<region>':  # it's a region - save the following and add group data
			self.regions.append(SFZ_region())
			self.curr += 1
			if self.in_group:
				self.regions[self.curr].update(self.group)

			self.in_region = True
		else:  # this should be the assignment
			segments = chunk.split('=')
			if len(segments) != 2:
				# maybe, we can just append this data to the previous chunk
				if self.last_chunk is not None:
					self.regions[self.curr][self.last_chunk[0]] += ' ' + segments[0]
					segments = (self.last_chunk[0], self.regions[self.curr][self.last_chunk[0]])
				else:
					print('Ambiguous spaces in SFZ file:', self.filename)
					sys.exit(1)
			if self.in_region:
				self.regions[self.curr][segments[0]] = segments[1]
			elif self.in_group:
				self.group[segments[0]] = segments[1]
			self.last_chunk = segments


def magic(filename, cwd, tempdir):
	start = timeit.default_timer()
	instrument = SFZ_instrument(cwd + filename, cwd, tempdir)

	fp = open(cwd + filename[:-4] + '.temp.xi', 'w')
	# create xi file
	fp.write(struct.pack('<21s22sb20sh',
		'Extended Instrument: ', (filename[:-4] + ' ' * 22)[:22], 0x1a,
		pad_name(VERSION, 20), 0x0102))

	notes_samples = [0 for i in range(96)]

	overlapping = []
	ignored = []


	if len(instrument.regions) >= 16:
		print('Too many samples in file:', filename, ' (no more than 16 samples supported)')
		instrument.regions = instrument.regions[:16]

	i = 0
	for region in instrument.regions:
		for note in region['notes']:
			if note < len(notes_samples) and note > -1:
				if notes_samples[note] != 0:
					overlapping.append(NOTES[note])
				notes_samples[note] = i
			else:
				ignored.append(NOTES[note])
		i += 1

	if len(overlapping) > 0:
		print('/' * 80)
		print(wrap('Notice: some regions are overlapping and would be overwritten', 80))
		print(wrap(str(overlapping), 80))
		print('/' * 80)
	if len(ignored) > 0:
		print('/' * 80)
		print(wrap('Notice: some notes are out of range and ignored', 80))
		print(wrap(str(ignored), 80))
		print('/' * 80)

	fp.write(struct.pack('<96b', *(notes_samples)))

	stt = 50  # seconds-to-ticks converter

	# volume envelope
	volume_points = 0
	volume_ticks = 0
	volume_envelope = []
	if 'ampeg_attack' not in region:
		volume_level = 0x40
	else:
		volume_level = 0
	vol_sustain_point = 0

	#fp.write(struct.pack('<h', volume_ticks))
	volume_envelope.append(volume_ticks)
	if 'ampeg_delay' in region:
		volume_ticks += float(region['ampeg_delay']) * stt
		volume_points += 1
		volume_level = 0

		#fp.write(struct.pack('<h', volume_level))
		volume_envelope.append(volume_level)
		#fp.write(struct.pack('<h', volume_ticks))
		volume_envelope.append(volume_ticks)

	if 'ampeg_start' in region:
		volume_level = int(float(region['ampeg_start']) / 100 * stt)

	if 'ampeg_attack' in region:
		volume_ticks += int(float(region['ampeg_attack']) * stt)

	#fp.write(struct.pack('<h', volume_level))
	volume_envelope.append(volume_level)
	volume_points += 1

	if 'ampeg_hold' in region:
		volume_ticks += int(float(region['ampeg_hold']) * stt)
	else:
		volume_level = 0x40
	#fp.write(struct.pack('<h', volume_ticks))
	volume_envelope.append(volume_ticks)
	#fp.write(struct.pack('<h', volume_level))
	volume_envelope.append(volume_level)
	volume_points += 1

	if 'ampeg_decay' in region:
		volume_ticks += int(float(region['ampeg_decay']) * stt)
		#fp.write(struct.pack('<h', volume_ticks))
		volume_envelope.append(volume_ticks)

		if 'ampeg_sustain' in region:
			#fp.write(struct.pack('<h', int(float(region['ampeg_sustain']) / 100 * stt)))
			volume_envelope.append(int(float(region['ampeg_sustain']) / 100 * stt))
		else:
			#fp.write(struct.pack('<h', 0))
			volume_envelope.append(0)

		volume_points += 1

	if 'ampeg_sustain' in region:
		volume_level = int(float(region['ampeg_sustain']) / 100 * stt)
		#fp.write(struct.pack('<h', volume_ticks))
		volume_envelope.append(volume_ticks)
		#fp.write(struct.pack('<h', volume_level))
		volume_envelope.append(volume_level)
		volume_points += 1
		vol_sustain_point = volume_points - 1

	if 'ampeg_release' in region:
		volume_ticks += int(float(region['ampeg_release']) * stt)
		volume_level = 0x0
		#fp.write(struct.pack('<h', volume_ticks))
		volume_envelope.append(volume_ticks)
		#fp.write(struct.pack('<h', volume_level))
		volume_envelope.append(volume_level)
		volume_points += 1

	if volume_ticks > 512:
		for i in range(int(len(volume_envelope) / 2)):
			volume_envelope[2 * i] = int(volume_envelope[2 * i] * 512 / volume_ticks)
		print('/' * 80)
		print('Too long envelope:', volume_ticks, 'ticks, shrinked to 512')
		print('/' * 80)

	fp.write(struct.pack('<{0}h'.format(2 * volume_points), *(volume_envelope)))
	fp.write(struct.pack('<{0}h'.format(2 * (12 - volume_points)), *(0 for i in range(2 * (12 - volume_points)))))
	#envelope = [0, 64, 4, 50, 8, 36, 13, 28, 20, 22, 33, 18, 47, 14, 62, 8, 85, 4, 161, 0, 100, 0, 110, 0]
	#fp.write(struct.pack('<24h', *(envelope)))
	fp.write(struct.pack('<24h', *(0 for i in range(24))))  # panning envelope

	fp.write(struct.pack('<b', volume_points))
	fp.write(struct.pack('<b', 0))

	fp.write(struct.pack('<b', vol_sustain_point))

	fp.write(struct.pack('<5b', *(0 for i in range(5))))

	volume_type = 0
	if volume_points > 0:
		volume_type += 0b1
	if vol_sustain_point > 0:
		volume_type += 0b10

	fp.write(struct.pack('<b', volume_type))
	fp.write(struct.pack('<b', 0))

	# vibrato type/sweep/depth/rate
	fp.write(struct.pack('<4b', *(0 for i in range(4))))

	# envelope data
	#fp.write(struct.pack('<b'))

	fp.write(struct.pack('<h', 0))  # volume fadeout
	fp.write(struct.pack('<22b', *(0 for i in range(22))))  # extended data
	fp.write(struct.pack('<h', len(instrument.regions)))  # number of samples

	for region in instrument.regions:
		fp.write(struct.pack('<i', region['sample_bittype'] * region['sample_length']))  # sample length
		fp.write(struct.pack('<2i', 0, 0))  # sample loop start and end
		# volume
		if 'volume' in region:
			fp.write(struct.pack('<B', math.floor(255 * math.exp(float(region['volume']) / 10) / math.exp(0.6))))	# 'cause volume is in dB
		else:
			fp.write(struct.pack('<B', 255))

		fp.write(struct.pack('<b', int(region['tune'])))  # finetune (signed!)
		fp.write(struct.pack('<b', region['sample_type']))  # sample type

		#panning (unsigned!)
		if 'pan' in region:
			fp.write(struct.pack('<B', (float(region['pan']) + 100) * 255 / 200))
		else:
			fp.write(struct.pack('<B', 128))

		if 'pitch_keycenter' in region:
			fp.write(struct.pack('<b',
			 NOTES.index(region['pitch_keycenter'])
			 - NOTES.index('c5')))	# relative note - transpose c4 ~ 00
		else:
			fp.write(struct.pack('<b',
			 NOTES.index(region['lokey'])
			 - NOTES.index('c5')))	# relative note - transpose c4 ~ 00

		sample_name = pad_name(os.path.split(region['sample_path'])[1], 22)

		fp.write(struct.pack('<b', len(sample_name.strip(' '))))
		fp.write(struct.pack('<22s', sample_name))

	for region in instrument.regions:
		df = open(region['delta_sample'], 'r')
		fp.write(df.read())
		df.close()

		os.remove(region['delta_sample'])
		region = {}
		del region

	print(len(instrument.regions), 'samples')
	print(int(fp.tell() / 1024), 'kB written in file "', filename, '" during', timeit.default_timer() - start, 'seconds')
	fp.close()
	instrument = {}
	del instrument
	os.rename(cwd + filename[:-4] + '.temp.xi', cwd + filename[:-4] + '.xi')


def main(argv):
	if '--force' in argv:
		force = True
		del argv[argv.index('--force')]
	else:
		force = False

	if len(argv) < 2:
		print('No input file specified')
		return 2

	try:
		cwd = os.getcwd() + '/'
		tempdir = tempfile.mkdtemp()

		start_time = timeit.default_timer()
		converted = 0
		for arg in argv[1:]:
			if not os.path.exists(cwd + arg[:-4] + '.xi') or force:
				print('-' * 80)
				print('Converting "', arg, '"')
				print('-' * 80)
				magic(arg, cwd, tempdir)
				converted += 1
			else:
				print('File', arg, 'is already converted!')

		print('')
		print(converted, 'files converted in', timeit.default_timer() - start_time, 'seconds')
	finally:
		try:
			shutil.rmtree(tempdir)	# delete directory
		except OSError as e:
			if e.errno != 2:  # code 2 - no such file or directory
				raise


if __name__ == '__main__':
	sys.exit(main(sys.argv))

