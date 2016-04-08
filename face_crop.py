#!/usr/bin/env python
"""
Crops short clips of people's faces out of a video when they're likely
speaking.

Accomplishes this by looking for when two conditions are met for a given
sequence of frames:

    1. There is exactly one face detected in all of the frames, in roughly the
       same position.
    2. Most of the corresponding audio frames have a max amplitude over some
       threshold.

When these conditions are met over some sequence of frames, the person's face
will be cropped out and the resulting video will be written to a video file.

Of course, this assumes that the source videos consist of mostly people
speaking. (news, speeches, vlogs, etc.) Don't expect this to work on videos
that don't meet this assumption.

Uses OpenCV for Haar-cascade face detection and ffmpeg for reading/writing
videos.

"""

import argparse, gc, os, subprocess, re, sys

import cv2
import numpy as np
import numpy.fft
import scipy.io.wavfile
import scipy.signal


class VideoReader:

    def __init__(self, video_path):
        """
        Constructor for a VideoReader object.

        Args:
            video_path (str): Path to the video to read.
        """

        self.video_path = video_path
        self.current_frame = 0

        self.proc = None

        self.get_info()
        self.load_audio()
        self.open_stream()


    def __del__(self):
        """
        Destructor for a VideoReader object - Kill the ffmpeg process if it's
        not already.
        """

        if self.proc is not None:
            self.proc.terminate()


    def get_info(self):
        """
        Gets information about a video (framerate, sampling rate, etc.) from
        ffmpeg.
        """

        # Summon ffmpeg to spit info about the video
        command = ['ffmpeg', '-i', self.video_path]
        proc = subprocess.Popen(command, stderr=subprocess.PIPE)
        output = str(proc.stderr.read())
        split_output = output.split(',')
        proc.terminate()

        # Extract information from ffmpeg's output
        fps = [x for x in split_output if 'fps' in x][0]
        hz = [x for x in split_output if 'Hz' in x][0]
        size = self.get_info.size_regex.findall(output)[1]

        self.frame_rate = float(fps.split()[0])
        self.sampling_rate = int(hz.split()[0])
        self.frame_size = int(size.split('x')[0]), int(size.split('x')[1])

    get_info.size_regex = re.compile('[0-9]+x[0-9]+')


    def load_audio(self):
        """
        Loads the audio from the video.
        """

        video_file = os.path.basename(self.video_path)
        video_name = os.path.splitext(video_file)[0]
        wav_file = 'tmp.{}.wav'.format(video_name)

        command = 'ffmpeg -y -i {} -vn {} 2> /dev/null'.format(
                self.video_path, wav_file)
        retval = os.system(command)

        if retval != 0:
            raise RuntimeError("Error extracting audio!")

        # Read in audio
        rate, self.audio = scipy.io.wavfile.read(wav_file)
        os.remove(wav_file)
        if rate != self.sampling_rate:
            raise RuntimeError("Sampling rate in .wav does not match video!")

        # Squash to mono
        if len(self.audio.shape) > 1:
            mean_audio = np.mean(self.audio, axis=1)
        else:
            mean_audio = self.audio

        self.audio_norm = mean_audio / np.max(np.abs(mean_audio))


    def open_stream(self):
        """
        Opens the video stream from ffmpeg.
        """

        command = ['ffmpeg',
            '-i', self.video_path,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)


    def next_frame(self):
        """
        Get the next frame from the video stream.

        Returns:
            numpy.ndarray, the next frame in the video, or None, when there are
            no frames left.
        """

        width, height = self.frame_size
        num_bytes = width*height*3

        raw = self.proc.stdout.read(num_bytes)

        if len(raw) == num_bytes:
            # Get video frame from ffmpeg
            video_frame = np.fromstring(raw, dtype='uint8')
            video_frame = video_frame.reshape((height, width, 3))
            self.proc.stdout.flush()

            # Get relevant audio frame
            audio_size = np.floor(self.sampling_rate/self.frame_rate)
            a = self.current_frame * audio_size
            b = a + audio_size
            audio_frame = self.audio_norm[a:b]

            if audio_frame.size != b-a:
                self.proc.terminate()
                self.proc = None
                return None, None

            self.current_frame += 1

            return video_frame, audio_frame
        else:
            self.proc.terminate()
            self.proc = None
            return None, None


class FaceCropper:

    def __init__(self, input_dir, output_dir,
            classifier_file='haarcascade_frontalface_default.xml',
            min_clip_length = 2.0, max_clip_length = 5.0,
            audio_threshold = 0.2, allowed_silence = 0.5):
        """
        Constructor for a FaceCropper object.

        Args:
            input_dir (str): Input directory.
            output_dir (str): Output directory.
        Args (optional):
            classifier_file (str): Name of the file to load for the
                cv2.CascadeClassifier object for face detection.
            min_clip_length (float): Min length of a clip. (in seconds)
            max_clip_length (float): Max length of a clip. (in seconds)
            audio_threshold (float): Amplitude threshold for audio to exceed.
            allowed_silence (float): Max length allowed of audio under the
                amplitude threshold.
        """

        # Validate parameters
        if (min_clip_length >= max_clip_length or min_clip_length <= 0.0
                or max_clip_length <= 0):
            raise RuntimeError("Invalid min/max clip lengths!")
        if audio_threshold < 0.0 or 1.0 < audio_threshold:
            raise RuntimeError("Audio threshold must be between 0 and 1")
        if allowed_silence < 0.0:
            raise RuntimeError("Allowed silence must be a positive value")

        self.min_clip_length = min_clip_length
        self.max_clip_length = max_clip_length
        self.audio_threshold = audio_threshold
        self.allowed_silence = allowed_silence

        # Validate input directory
        if not os.path.isdir(input_dir):
            raise RuntimeError("{} is not a directory!".format(input_dir))

        # Create output directory if it doesn't exist
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

        self.input_dir       = input_dir
        self.output_dir      = output_dir
        self.face_classifier = cv2.CascadeClassifier(classifier_file)

        if self.face_classifier.empty():
            raise RuntimeError("Unable to load {}!".format(classifier_file))


    def process(self):
        """
        Processes every .mp4 file in the input directory.
        """

        def find_all_videos(base_dir):
            """ Recursively find all .mp4 files in a directory. """

            video_files = list()

            for item in os.listdir( os.path.join(self.input_dir, base_dir) ):
                item_path = os.path.join(base_dir, item)
                full_path = os.path.join(self.input_dir, item_path)
                if os.path.isfile(full_path):
                    if os.path.splitext(item)[1] == '.mp4':
                        video_files.append(item_path)
                elif os.path.isdir(full_path):
                    video_files.extend( find_all_videos(item_path) )

            return video_files

        video_files = find_all_videos('')

        for video_file in video_files:
            self.segment_video(video_file)


    def segment_video(self, video_file):
        """
        Segments and crops the video based on face detection and audio
        onset/offset results.

        Args:
            video_file (str): Video to process.
                (assumed to be in self.input_dir)
        """

        print("Processing {}".format(video_file))

        video_path = os.path.join(self.input_dir, video_file)

        print("\tOpening video reader...")

        video_reader = VideoReader(video_path)

        # State parameters

        width, height = video_reader.frame_size

        num_clips   = 0  # The number of clips that have been created
        start_frame = 0  # Start frame of the current clip

        min_length     = np.round(video_reader.frame_rate*self.min_clip_length)
        max_length     = np.round(video_reader.frame_rate*self.max_clip_length)
        length_silence = np.round(video_reader.frame_rate*self.allowed_silence)
        num_silence = 0

        recording = False # Whether or not we're retaining frames

        frames = list()
        positions = list()

        last_pos = None

        print("\tBeginning scan...")

        video_frame, audio_frame = video_reader.next_frame()
        while video_frame is not None:

            # Check if there's a face in the video frame

            has_face = False
            has_audio = False

            gray = cv2.cvtColor(video_frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 1:
                has_face = True
                (x,y,w,h) = faces[0]

                if last_pos is not None:
                    if (np.abs(x-last_pos[0]) > width/10 or
                            np.abs(y-last_pos[1]) > height/10):
                        has_face = False

            # Check if the audio is over a given threshold

            amplitude = np.max(np.abs(audio_frame))

            if amplitude >= self.audio_threshold:
                num_silence = 0
                has_audio = True
            else:
                num_silence += 1
                if recording and num_silence < length_silence:
                    has_audio = True

            # If both conditions are met, record the current frame

            if has_face and has_audio:
                if not recording:
                    recording = True
                    start_frame = video_reader.current_frame-1

                positions.append( (x,y,w,h) )
                last_pos = x,y
                frames.append(video_frame)

                # If the number of recorded frames is over the max limit,
                # write now and reset

                if video_reader.current_frame-start_frame >= max_length:

                    a = start_frame*audio_frame.size
                    b = video_reader.current_frame*audio_frame.size-1

                    frames = self.crop_frames(frames, positions,
                            video_reader.frame_rate)
                    self.write_video(frames, video_reader.audio[a:b],
                            video_reader.frame_rate,
                            video_reader.sampling_rate, video_file, num_clips)
                    num_clips += 1

                    del frames[:]
                    del positions[:]

                    recording = False
                    num_silence = 0
                    last_pos = None

            # If conditions aren't met, but we're recording...

            elif recording:
                # Write if we're above the minimum limit
                if video_reader.current_frame-start_frame > min_length:
                    a = start_frame*audio_frame.size
                    b = video_reader.current_frame*audio_frame.size-1

                    frames = self.crop_frames(frames, positions,
                            video_reader.frame_rate)
                    self.write_video(frames, video_reader.audio[a:b],
                            video_reader.frame_rate,
                            video_reader.sampling_rate, video_file, num_clips)
                    num_clips += 1

                del frames[:]
                del positions[:]

                recording = False
                num_silence = 0
                last_pos = None

            video_frame, audio_frame = video_reader.next_frame()


    def crop_frames(self, frames, positions, fps):
        """
        Crops regions out of a set of frames, making sure that they are all
        the same size.

        Args:
            frames (list<numpy.ndarray>): Frames to crop from.
            positions (list<tuple>): x,y,w,h positions to crop.
        Returns:
            list<numpy.ndarray>, cropped frames.
        """

        max_w = 0
        max_h = 0

        for x,y,w,h in positions:
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h

        # Fix x,y coords so they stretch out to fit the max width, height

        fixed_xs = list()
        fixed_ys = list()

        for idx, pos in enumerate(positions):
            x,y,w,h = pos

            diff_w = max_w-w
            diff_h = max_h-h

            fixed_x = x-np.floor(diff_w/2)
            fixed_y = y-np.floor(diff_h/2)

            fixed_xs.append(fixed_x)
            fixed_ys.append(fixed_y)

        # Low-pass x,y coords to stop crop from shaking so much
        nyq = 0.5 * fps # Nyquist rate
        cutoff = 1/nyq # Cutoff at 1Hz
        b, a = scipy.signal.butter(8, cutoff)

        fixed_xs = np.round( scipy.signal.filtfilt(b, a, fixed_xs) )
        fixed_ys = np.round( scipy.signal.filtfilt(b, a, fixed_ys) )

        # Get crops

        new_frames = list()

        for i in range(0, len(fixed_xs)):
            x = fixed_xs[i]
            y = fixed_ys[i]

            new_frames.append( frames[i][y:y+max_h,x:x+max_w,:] )

        return new_frames


    def write_video(self, frames, audio, frame_rate, sampling_rate,
            video_file, num_video):
        """
        Writes a set of frames and audio to a video by piping data into ffmpeg.

        Args:
            frames (list<numpy.ndarray>): Video frames to write.
            audio (numpy.ndarray): Audio stream to write. Will be written to a
                file first using scipy.
            frame_rate (float): Frame rate of the video.
            sampling_rate (int): Sampling rate of the audio.
            video_path (str): Path to the source video.
            num_video (int): The number to label this video with.
        """

        # Get name of video
        video_name, video_ext = os.path.splitext(video_file)
        video_name = os.path.basename(video_name)

        # Extract directory and create it in output, if needd
        video_dir  = os.path.dirname(video_file)

        if not os.path.exists( os.path.join(self.output_dir, video_dir) ):
            os.makedirs( os.path.join(self.output_dir, video_dir) )

        # Write audio to temp file
        audio_file = 'tmp.{}.wav'.format(video_name)
        scipy.io.wavfile.write(audio_file, sampling_rate, audio)

        output_path = os.path.join(self.output_dir, video_dir,
                '{}-{:03}{}'.format(video_name, num_video, video_ext))

        print("\t\tWriting {} ({} frames)".format(output_path, len(frames)))

        height, width, _ = frames[0].shape

        command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '{}x{}'.format(width, height),
            '-pix_fmt', 'rgb24',
            '-r', str(frame_rate),
            '-i', '-',
            '-an',
            '-vcodec', 'h264',
            '-i', audio_file,
            '-c:a', 'aac',
            output_path
        ]
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE,
                stderr=subprocess.PIPE)
        for frame in frames:
            frame.tostring()
            pipe.stdin.write( frame.tostring() )
        pipe.stdin.close()
        if pipe.stderr is not None:
            pipe.stderr.close()
        pipe.wait()
        del pipe

        os.remove(audio_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
            "Crops short clips of people's faces out of a video."
    )
    parser.add_argument('-data', default='data', help=
            "Directory where data lives."
    )
    parser.add_argument('-output', default='output', help=
            "Where resulting trims should be stored."
    )
    parser.add_argument('-min_clip_length', type=float, default=2.0, help=
            "Min length of a clip. (in seconds)"
    )
    parser.add_argument('-max_clip_length', type=float, default=5.0, help=
            "Max length of a clip. (in seconds)"
    )
    parser.add_argument('-audio_threshold', type=float, default=0.2, help=
            "Normalized amplitude threshold for audio to exceed."
    )
    parser.add_argument('-allowed_silence', type=float, default=0.5, help=
            "Max length allowed of audio under the amplitude threshold."
    )
    args = parser.parse_args()

    fc = FaceCropper(
        args.data,
        args.output,
        min_clip_length = args.min_clip_length,
        max_clip_length = args.max_clip_length,
        audio_threshold = args.audio_threshold,
        allowed_silence = args.allowed_silence,
    )
    fc.process()


