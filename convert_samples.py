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

from __future__ import (division, print_function)

import sys
import os

import subprocess
import argparse
import tempfile


SAMPLE_EXTENSIONS = [
	'.wav',
	'.flac',
	'.ogg',
	'.oga',
	'.mp3',
	'.aac',
	'.m4a',
]


def main(argv):
	parser = argparse.ArgumentParser(prog=argv[0], description='Convert .flac or stereo .wav samples to mono .wav format')
	parser.add_argument('-d', '--output-dir', help='set output directory (default: current directory)')
	parser.add_argument('--sox-cmd', help='set sox executable name/path (default: sox)', default='sox')

	parser.add_argument('sample_dir', help='sample directory')
	options = parser.parse_args(argv[1:])

	if not options.output_dir:
		options.output_dir = os.path.abspath(os.getcwd())
	else:
		options.output_dir = os.path.abspath(options.output_dir)

	options.sample_dir = os.path.abspath(options.sample_dir)
	if not os.path.isdir(options.sample_dir):
		print('ERROR: Invalid sample directory')
		return 2

	if not os.path.isdir(options.output_dir):
		os.makedirs(options.output_dir)

	for root, dirs, files in os.walk(options.sample_dir):
		for name in files:
			root_, ext = os.path.splitext(name)
			if ext not in SAMPLE_EXTENSIONS:
				continue

			full_name = os.path.join(root, name)

			print('Converting:', name)

			out_options = []
			effects = []

			# Get the number of channels of the sample
			cmd = [options.sox_cmd, '--info', '-c', full_name]
			stdoutdata = subprocess.check_output(cmd)
			if stdoutdata.strip() != '1':
				effects.extend(['channels', '1'])

			# Get the number of bits per sample of the sample
			cmd = [options.sox_cmd, '--info', '-b', full_name]
			stdoutdata = subprocess.check_output(cmd)
			bits_per_sample = int(stdoutdata.strip())
			if bits_per_sample > 8:
				out_options.extend(['-b', '16', '-e', 'signed-integer'])
			else:
				out_options.extend(['-b', '8', '-e', 'unsigned-integer'])

			# Create the output directory if it doesn't exist
			out_dir = os.path.normpath(''.join([options.output_dir, root[len(options.sample_dir):]]))
			if not os.path.isdir(out_dir):
				os.makedirs(out_dir)

			# Although the mktemp() function is labeled as insecure, it is good enough for the purposes of this script
			temp_name = tempfile.mktemp(suffix='.wav', dir=out_dir)

			# Convert the sample to a temporary file
			cmd = [options.sox_cmd, full_name]
			cmd.extend(out_options)
			cmd.append(temp_name)
			cmd.extend(effects)
			subprocess.check_call(cmd)

			# Rename the temporary file to the correct name
			out_name = os.path.join(out_dir, ''.join([root_, '.wav']))

			if os.path.exists(out_name):
				os.remove(out_name)

			os.rename(temp_name, out_name)

	print('')
	print('Done.')


if __name__ == '__main__':
	sys.exit(main(sys.argv))

