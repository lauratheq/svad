#!/usr/bin/python3

# NAME
#   SVAD - Simple Voice Activity Detector
#
# SYNOPSIS
#   ./svad.py [--sample_path] [--sample_rate] [--buffer_size] [--error_margin] [--log_level] [--verbose] [--help]
#
# DESCRIPTION
#   This script 
#
# EXAMPLE:
#   ./svad.py --sample_path ./samples --sample_rate 22050 --buffer_size 1024 --error_margin 5
#
# OPTIONS
#   -p|--sample_path    path to the samples directory
#   -r|--sample_rate    resolution of the audio stream
#   -b|--buffer_size    size of the buffer
#   -e|--error_margin   the error margin in percentile
#   -l|--log_level      the displayed information this script generates
#   log levels: debug, info, warning, error, critical
#   -v|--verbose        overwrites the log level to debug
#   -h|--help           displays the help message
#
# LEGAL NOTE
#   Written and maintained by Laura Herzog (laura-herzog@outlook.com)
#   Concept and ideas by Vanessa Jahn-Ruhland, Thomas Davies and
#   Laura Herzog
#   Permission to copy and modify is granted under the GPLv3 license
#   Project Information: https://github.com/lauratheq/svad/

# import our own config which defines the basic settings
# throughout this project, all the config variables can be
# overwritten by calling this script with according operators
#
# you can access these properties:
#   config.SAMPLE_RATE  int     ticks per second
#   config.BUFFER_SIZE  int     amount of ticks are stored in the buffer
#   config.ERROR_MARGIN int     percentile of the error margin in which
#                               the system operates
#   config.LOG_LEVEL    str     the defined log level of the config package
#                               this can be overwritten with the option
#                               --log_level calling this script
import config

# import python standard libraries
import logging
import os
import librosa
import math
import numpy as np
import pyaudio
import sys
import getopt
from glob import glob
import pymsgbox

class SVAD():
    '''
    This is the main class of the SVAD system.
    As it works from top to bottom:
        * parsing the operators and set the basic settings
        * converting the samples into the needed format
        * starting the microphone input stream
        * starting an infinite loop
            * converting the data of the input stream to
              the needed format
            * comparing the chunks with each other using
              the eucledean distance method
            * break the loop if there are enough matches
    '''

    # the relative path to the samples directory
    sample_path = 'samples'

    # the resolution of the samples
    sample_rate = config.SAMPLE_RATE

    # the buffer size which also represents the chunk size
    buffer_size = config.BUFFER_SIZE

    # the error margin as integer
    error_margin = config.ERROR_MARGIN

    # holder for the maximum existing chunks to compare
    max_chunks = 0

    # holder for the max_chunks + error_margin
    max_chunks_pls = 0

    # holder for the max_chunks - error_margin
    max_chunks_mns = 0

    # the logger instance holder
    logger = None

    # the current set log level
    log_level = config.LOG_LEVEL

    # currently available log levels
    log_levels = {
        "NOTSET": 0,
        "DEBUG": 10,
        "INFO": 20,
        "WARN": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }

    # flag if we want a verbose output
    verbose = False

    # the data holder for the input stream
    stream = None

    # the pyAudio instance
    p = None

    # the stack dict for the chunks of each sample
    samples_chunk_stack = {}

    # the stack list for the chunks of the input stream
    stream_chunk_stack = []

    def __init__(self):
        '''
        Initialzes the class with all its needed parameters

        Parameters:
            self (obj): the class object

        Returns:
            void
        '''

        # set the absolute directory of the default samples path
        if os.path.isabs(self.sample_path) == False:
            self.sample_path = os.path.abspath(self.sample_path)

        # parse the operators and set the arguments
        opts, args = getopt.getopt(sys.argv[1:], "hvs:l:e:r:b:", ["help", "sample_path=", "log_level=", "verbose", "error_margin=", "buffer_size=", "sample_rate="])
        for operator, argument in opts:

            # check if we want to overwrite the sample path
            if operator in ("-s", "--sample_path"):
                # make path absolute
                if os.path.isabs(argument) == False:
                    argument = os.path.abspath(argument)

                # determine if the samples path exists
                if os.path.isdir(argument):
                    self.sample_path = argument
                else:
                    print(f"Path to samples not found: {sample_path}")
                    sys.exit()
            # check if we need to overwrite the sample rate
            elif operator in ("-r", "--sample_rate"):
                self.sample_rate = argument
            # same for the buffer size
            elif operator in ("-b", "--buffer-size"):
                self.buffer_size = argument
            # of course the error margin
            elif operator in ("-e", "--error_margin"):
                self.error_margin = argument
            # set the log level
            elif operator in ("-l", "--log_level"):
                self.log_level = argument.upper()
            # check if we need the verbose flag and change the log level
            elif operator in ("-v", "--verbose"):
                self.log_level = 'DEBUG'
                self.verbose = True
            # check if we need to display the help
            elif operator in ("-h", "--help"):
                self.display_help()
                sys.exit()
            else:
                pass

        # initialize logger
        self.init_logger()
        self.logger.debug("Logger initialized")
        self.logger.info("Starting SVAD with following arguments:")
        self.logger.info(f"    sample_path: {self.sample_path}")
        self.logger.info(f"    log_level: {self.log_level}")
        self.logger.info(f"    verbose: {self.verbose}")
        self.logger.info(f"    sample_rate: {self.sample_rate}")
        self.logger.info(f"    buffer_size: {self.buffer_size}")
        self.logger.info(f"    error_margin: {self.error_margin}")

    def run(self):
        '''
        Batch function to start all the needed features if this script.
        The batch:
            1. converts the samples to the needed format
            2. starts the stream for the audio input via mircophone
            3. converts the input stream to the needed format
            4. starts the loop to compare the stream with the samples

        Parameters:
            self (obj): the class object

        Returns:
            void
        '''

        # we start by converting the samples to the needed data format
        # which is described in the actual method
        self.convert_samples()

        # load the microphone stream
        self.stream = self.load_microphone_stream()

        # start the loop
        self.logger.info("Starting comparing loop")
        while(True):
            # overwrite `data` with the data in the buffer
            data = self.stream.read(self.buffer_size)

            # if the chunk buffer exceedes the max chunk amount
            # of our samples we we remove the first chunk of
            # the microphone buffer. This is because we need to
            # save some memory
            if len(self.stream_chunk_stack) >= self.max_chunks:
                self.stream_chunk_stack.pop(0)

            # convert the binary data into understandable data
            buffer_chunk_data = np.frombuffer(data, dtype=np.float32)

            # make every datapoint positive to make it easier calculating
            buffer_chunk_data = np.abs(buffer_chunk_data)

            # sum the data
            buffer_chunk_sum = np.sum(buffer_chunk_data)

            # append the buffer to the stack
            self.stream_chunk_stack.append(buffer_chunk_sum)

            # we donot start the compare algorythm until we have enough
            # chunks filled with our data
            if len(self.stream_chunk_stack) < self.max_chunks:
                continue
            
            # the buffer is filled, we now can compare the stuff
            if self.compare_data() == True:
                self.logger.info("Match has been found")
                self.msg_box('Match found', 'The phrase "OK BOOMER" has been detected. Microphone is now offline.')
                break

    def compare_data(self):
        '''
        Compares the data

        Parameters:
            self (obj): the class object

        Returns:
            bool
        '''
        # walk each sample and load the data
        for sample_name in self.samples_chunk_stack:
            # get the data of the sample
            sample_data = self.samples_chunk_stack[sample_name]

            # walk the input stream chunk data and compare with the sample data
            i = 0
            chunk_match_hits = 0
            for stream_sum in self.stream_chunk_stack:
                # get the data to compare
                particular_sample_chunk_data = sample_data.get(i)
                particular_stream_chunk_data = self.stream_chunk_stack[i]
                if particular_sample_chunk_data and particular_stream_chunk_data:
                    # check if the values are within the error margin
                    if particular_sample_chunk_data["mns"] <= particular_stream_chunk_data <= particular_sample_chunk_data["pls"]:
                        chunk_match_hits += 1
                i += 1

            # check how many chunk match hits we have for this sample
            # we respect the error margin here as well
            self.logger.debug(f"Chunk matches: {chunk_match_hits}")

            # compare the amount of chunk matches with the error margin
            if self.max_chunks_mns <= chunk_match_hits <= self.max_chunks_pls:
                return True
            else:
                return False

    def convert_samples(self):
        '''
        Loads the samples of the given path and calculates the sum
        of all data points. It also adds and removes the error margin
        and saves it in the samples_chunk_stack

        Parameters:
            self (obj): the object class

        Returns:
            void
        '''

        # load the sample files via glob
        self.logger.info(f"Loading sample files from {self.sample_path}")
        sample_files = glob(self.sample_path + "/sample-*.wav")

        # walk the sample files to convert them to the needed data format
        for sample_file in sample_files:
            sample_file_name = os.path.basename(sample_file)
            self.logger.info(f"Loading {sample_file_name}")

            # add the key to the chunk stack so we can easily identify
            # the dataset
            self.samples_chunk_stack[sample_file] = []

            # load the dataset 
            data_points, sr = librosa.load(sample_file)
            data_shape = data_points.shape

            # dividing the dataset into chunks the size of the defined buffer
            amount_of_chunks = math.floor(data_shape[0] / self.buffer_size )

            # set the maximum amount of chunks so we can calculate the
            # needed amount of chunks within the error margin
            if amount_of_chunks > self.max_chunks:
                self.max_chunks = amount_of_chunks
                max_chunks_five = int(self.max_chunks) * int(self.error_margin) / 100
                self.max_chunks_pls = self.max_chunks + max_chunks_five
                self.max_chunks_mns = self.max_chunks - max_chunks_five

            chunks = np.array_split(data_points, amount_of_chunks)

            self.logger.debug(f"debug info for {sample_file_name}")
            self.logger.debug(f"    shape data_points: {data_points.shape[0]}")
            self.logger.debug(f"    amount of chunks: {amount_of_chunks}")

            # gets the sum of each chunk and add/remove the
            # error margin to its sum
            i = 0
            chunks_data = {}
            for chunk in chunks:
                # make everything positive
                chunk = np.abs(chunk)

                # sum the array
                chunk_sum = np.sum(chunk)

                # calculate the error margin for this sample
                chunk_five = float(chunk_sum) * int(self.error_margin) / 100
                chunk_data = {
                    "sum": chunk_sum,
                    "pls": chunk_sum + chunk_five,
                    "mns": chunk_sum - chunk_five
                }
                chunks_data[i] = chunk_data
                i += 1

            # add the data to the stack
            self.samples_chunk_stack[sample_file] = chunks_data
            self.logger.debug(f"Loading {sample_file_name} done")
        self.logger.info("Samples loaded")

    def load_microphone_stream(self):
        '''
        Loads the microphone stream using pyaudio

        Parameters:
            self (obj): the class object

        Returns:
            stream (stream): the input stream of the microphone
        '''
        self.logger.info("Starting audio stream")

        # initialize PyAudio which captures the microphone
        self.p = pyaudio.PyAudio()

        # start the input stream
        stream = self.p.open(
            # format is the same as in the sample recoder
            # see record-samples.py
            format=pyaudio.paFloat32,
            
            # we only use mono, that's enough
            channels=1,

            # sample rate
            rate=self.sample_rate,

            # buffer size
            frames_per_buffer=self.buffer_size,

            # input flag for the microphone to use
            input=True
        )
        return stream

    def init_logger(self):
        '''
        initializes the logger and sets the log level

        Parameters:
            self (obj): the class object

        Returns:
            void
        '''

        # build a logger instance for our tool
        self.logger = logging.getLogger('svad')

        # set the log level according to our settings
        self.logger.setLevel(self.log_levels[self.log_level])

        # we are using a stream handler for imidiate output
        handler = logging.StreamHandler()

        # set the formater for our handler
        format = '%(asctime)s - %(levelname)s - %(filename)s - %(message)s'
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)

        # add our handler to the logger
        self.logger.addHandler(handler)

    def display_help(self):
        '''
        displays the help message

        Parameters:
          self (obj): the class object
    
        Returns:
          void
        '''
 
        print('usage: ./svad.py [--sample_path] [--sample_rate] [--buffer_size] [--error_margin] [--log_level] [--verbose] [--help]')

    def exit(self):
        '''
        Cleans up after itself and closes all streams

        Parameters:
            self (obj): the class object

        Returns:
            void
        '''
        # stop the audio input stream
        self.stream.stop_stream()
        
        # close the microphone input stream
        self.stream.close()

        # destroy the pyaudio instance
        self.p.terminate()

    def msg_box(self, title, content):
        '''
        Displays a small message box with an "OK" button

        Parameters:
            self (obj): the class object
            title (str): The title to be displayed in the msg box
            content (str): the inner content

        Returns:
            pymsg - the message box
        '''
        return pymsgbox.alert(content, title)

# We only need to start the system if this file is called
if __name__ == "__main__":
    try:
        # initializing the instance
        svad = SVAD()

        # start the comparing
        svad.run()
    except KeyboardInterrupt:
        try:
            # perform the exit of the svad system
            svad.exit()
            sys.exit(130)
        except SystemExit:
            # perform the exit of the svad system
            svad.exit()
            os._exit(130)
