"""Krauncher variant of tasks/lambdalabs/waveglow_ljs625_bs18.py (the neutral form, with the
source and the reduction it documents); the expected forecast and its
sources are in tasks/lambdalabs/waveglow_ljs625_bs18.json.

The task function holds the neutral file's code — the imports, the sections
copied verbatim from NVIDIA DeepLearningExamples (headed by file and lines,
licence headers kept) and the run — indented into one self-contained
function. Differences from the neutral form: the LJSpeech-1.1 archive and the
two file lists come through a registered data source (/data) instead of a
download; the DataLoader workers are forked (the dataset class is local to
the function and cannot be pickled). FUNC is the task, OPTIONS the arguments
of its @client.task decorator, KWARGS the arguments of the call.
tools/build_dle_waveglow.py rebuilds this file from a checkout of the source.
"""

OPTIONS = dict(
    timeout=3600,
    data="ljspeech-11-dle",
    pip=["librosa"],
)

# Data source the task reads (OPTIONS["data"]), registered on the account when
# missing: LJSpeech-1.1 (public domain, keithito.com) and the two file lists
# of the benchmark's code at the pinned commit.
DATA_SOURCES = [
    dict(name="ljspeech-11-dle", urls=[
        "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "https://raw.githubusercontent.com/LambdaLabsML/DeepLearningExamples/667536cc8138edbd236aed08dbafc8e77bcc7c20/PyTorch/SpeechSynthesis/Tacotron2/filelists/ljs_audio_text_train_subset_625_filelist.txt",
        "https://raw.githubusercontent.com/LambdaLabsML/DeepLearningExamples/667536cc8138edbd236aed08dbafc8e77bcc7c20/PyTorch/SpeechSynthesis/Tacotron2/filelists/ljs_audio_text_val_filelist.txt",
    ], size_gb=2.6),
]

KWARGS = dict(batch_size=18, epochs=2, segment_length=8000)


def train_waveglow(batch_size: int = 18, epochs: int = 2, segment_length: int = 8000):
    import argparse
    import os
    import time

    import numpy as np
    import librosa.util as librosa_util
    import torch
    import torch.nn.functional as F
    from librosa.filters import mel as librosa_mel_fn
    from librosa.util import pad_center, tiny
    from scipy.io.wavfile import read
    from scipy.signal import get_window
    from torch.autograd import Variable
    from torch.utils.data import DataLoader

    torch._C._jit_set_autocast_mode(False)  # waveglow/model.py:28

    # --- licence header: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/audio_processing.py:1-27 ---
    # *****************************************************************************
    #  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
    #
    #  Redistribution and use in source and binary forms, with or without
    #  modification, are permitted provided that the following conditions are met:
    #      * Redistributions of source code must retain the above copyright
    #        notice, this list of conditions and the following disclaimer.
    #      * Redistributions in binary form must reproduce the above copyright
    #        notice, this list of conditions and the following disclaimer in the
    #        documentation and/or other materials provided with the distribution.
    #      * Neither the name of the NVIDIA CORPORATION nor the
    #        names of its contributors may be used to endorse or promote products
    #        derived from this software without specific prior written permission.
    #
    #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    #  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    #  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    #  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
    #  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    #  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    #  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    #  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    #  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    #  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    #
    # *****************************************************************************


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/audio_processing.py:34-83 ---
    def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                         n_fft=800, dtype=np.float32, norm=None):
        """
        # from librosa 0.6
        Compute the sum-square envelope of a window function at a given hop length.

        This is used to estimate modulation effects induced by windowing
        observations in short-time fourier transforms.

        Parameters
        ----------
        window : string, tuple, number, callable, or list-like
            Window specification, as in `get_window`

        n_frames : int > 0
            The number of analysis frames

        hop_length : int > 0
            The number of samples to advance between frames

        win_length : [optional]
            The length of the window function.  By default, this matches `n_fft`.

        n_fft : int > 0
            The length of each analysis frame.

        dtype : np.dtype
            The data type of the output

        Returns
        -------
        wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
            The sum-squared envelope of the window function
        """
        if win_length is None:
            win_length = n_fft

        n = n_fft + hop_length * (n_frames - 1)
        x = np.zeros(n, dtype=dtype)

        # Compute the squared window at the desired length
        win_sq = get_window(window, win_length, fftbins=True)
        win_sq = librosa_util.normalize(win_sq, norm=norm)**2
        win_sq = librosa_util.pad_center(win_sq, size=n_fft)

        # Fill the envelope
        for i in range(n_frames):
            sample = i * hop_length
            x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
        return x


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/audio_processing.py:105-111 ---
    def dynamic_range_compression(x, C=1, clip_val=1e-5):
        """
        PARAMS
        ------
        C: compression factor
        """
        return torch.log(torch.clamp(x, min=clip_val) * C)


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/audio_processing.py:114-120 ---
    def dynamic_range_decompression(x, C=1):
        """
        PARAMS
        ------
        C: compression factor used to compress
        """
        return torch.exp(x) / C


    # --- licence header: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/stft.py:1-32 ---
    """
    BSD 3-Clause License

    Copyright (c) 2017, Prem Seetharaman
    All rights reserved.

    * Redistribution and use in source and binary forms, with or without
      modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice, this
      list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from this
      software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/stft.py:42-142 ---
    class STFT(torch.nn.Module):
        """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
        def __init__(self, filter_length=800, hop_length=200, win_length=800,
                     window='hann'):
            super(STFT, self).__init__()
            self.filter_length = filter_length
            self.hop_length = hop_length
            self.win_length = win_length
            self.window = window
            self.forward_transform = None
            scale = self.filter_length / self.hop_length
            fourier_basis = np.fft.fft(np.eye(self.filter_length))

            cutoff = int((self.filter_length / 2 + 1))
            fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                       np.imag(fourier_basis[:cutoff, :])])

            forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
            inverse_basis = torch.FloatTensor(
                np.linalg.pinv(scale * fourier_basis).T[:, None, :].astype(np.float32))

            if window is not None:
                assert(filter_length >= win_length)
                # get window and zero center pad it to filter_length
                fft_window = get_window(window, win_length, fftbins=True)
                fft_window = pad_center(fft_window, size=filter_length)
                fft_window = torch.from_numpy(fft_window).float()

                # window the bases
                forward_basis *= fft_window
                inverse_basis *= fft_window

            self.register_buffer('forward_basis', forward_basis.float())
            self.register_buffer('inverse_basis', inverse_basis.float())

        def transform(self, input_data):
            num_batches = input_data.size(0)
            num_samples = input_data.size(1)

            self.num_samples = num_samples

            # similar to librosa, reflect-pad the input
            input_data = input_data.view(num_batches, 1, num_samples)
            input_data = F.pad(
                input_data.unsqueeze(1),
                (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
                mode='reflect')
            input_data = input_data.squeeze(1)

            forward_transform = F.conv1d(
                input_data,
                Variable(self.forward_basis, requires_grad=False),
                stride=self.hop_length,
                padding=0)

            cutoff = int((self.filter_length / 2) + 1)
            real_part = forward_transform[:, :cutoff, :]
            imag_part = forward_transform[:, cutoff:, :]

            magnitude = torch.sqrt(real_part**2 + imag_part**2)
            phase = torch.autograd.Variable(
                torch.atan2(imag_part.data, real_part.data))

            return magnitude, phase

        def inverse(self, magnitude, phase):
            recombine_magnitude_phase = torch.cat(
                [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

            inverse_transform = F.conv_transpose2d(
                recombine_magnitude_phase.unsqueeze(-1),
                Variable(self.inverse_basis.unsqueeze(-1), requires_grad=False),
                stride=(self.hop_length,1),
                padding=(0,0))
            inverse_transform = inverse_transform.squeeze(-1)

            if self.window is not None:
                window_sum = window_sumsquare(
                    self.window, magnitude.size(-1), hop_length=self.hop_length,
                    win_length=self.win_length, n_fft=self.filter_length,
                    dtype=np.float32)
                # remove modulation effects
                approx_nonzero_indices = torch.from_numpy(
                    np.where(window_sum > tiny(window_sum))[0])
                window_sum = torch.autograd.Variable(
                    torch.from_numpy(window_sum), requires_grad=False)
                window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
                inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

                # scale by hop ratio
                inverse_transform *= float(self.filter_length) / self.hop_length

            inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
            inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

            return inverse_transform

        def forward(self, input_data):
            self.magnitude, self.phase = self.transform(input_data)
            reconstruction = self.inverse(self.magnitude, self.phase)
            return reconstruction


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/layers.py:68-112 ---
    class TacotronSTFT(torch.nn.Module):
        def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                     n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                     mel_fmax=8000.0):
            super(TacotronSTFT, self).__init__()
            self.n_mel_channels = n_mel_channels
            self.sampling_rate = sampling_rate
            self.stft_fn = STFT(filter_length, hop_length, win_length)
            mel_basis = librosa_mel_fn(
                sr=sampling_rate,
                n_fft=filter_length,
                n_mels=n_mel_channels,
                fmin=mel_fmin,
                fmax=mel_fmax
            )

            mel_basis = torch.from_numpy(mel_basis).float()
            self.register_buffer('mel_basis', mel_basis)

        def spectral_normalize(self, magnitudes):
            output = dynamic_range_compression(magnitudes)
            return output

        def spectral_de_normalize(self, magnitudes):
            output = dynamic_range_decompression(magnitudes)
            return output

        def mel_spectrogram(self, y):
            """Computes mel-spectrograms from a batch of waves
            PARAMS
            ------
            y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

            RETURNS
            -------
            mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
            """
            assert(torch.min(y.data) >= -1)
            assert(torch.max(y.data) <= 1)

            magnitudes, phases = self.stft_fn.transform(y)
            magnitudes = magnitudes.data
            mel_output = torch.matmul(self.mel_basis, magnitudes)
            mel_output = self.spectral_normalize(mel_output)
            return mel_output


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/utils.py:58-60 ---
    def load_wav_to_torch(full_path):
        sampling_rate, data = read(full_path)
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/utils.py:63-74 ---
    def load_filepaths_and_text(dataset_path, filename, split="|"):
        with open(filename, encoding='utf-8') as f:
            def split_line(root, line):
                parts = line.strip().split(split)
                if len(parts) > 2:
                    raise Exception(
                        "incorrect line format for file: {}".format(filename))
                path = os.path.join(root, parts[0])
                text = parts[1]
                return path,text
            filepaths_and_text = [split_line(dataset_path, line) for line in f]
        return filepaths_and_text


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/tacotron2_common/utils.py:77-82 ---
    def to_gpu(x):
        x = x.contiguous()

        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
        return x


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/waveglow/model.py:33-40 ---
    @torch.jit.script
    def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
        n_channels_int = n_channels[0]
        in_act = input_a + input_b
        t_act = torch.tanh(in_act[:, :n_channels_int, :])
        s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
        acts = t_act * s_act
        return acts


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/waveglow/model.py:43-91 ---
    class Invertible1x1Conv(torch.nn.Module):
        """
        The layer outputs both the convolution, and the log determinant
        of its weight matrix.  If reverse=True it does convolution with
        inverse
        """

        def __init__(self, c):
            super(Invertible1x1Conv, self).__init__()
            self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                        bias=False)

            # Sample a random orthonormal matrix to initialize weights
            W = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0]

            # Ensure determinant is 1.0 not -1.0
            if torch.det(W) < 0:
                W[:, 0] = -1 * W[:, 0]
            W = W.view(c, c, 1)
            W = W.contiguous()
            self.conv.weight.data = W

        def forward(self, z):
            # shape
            batch_size, group_size, n_of_groups = z.size()

            W = self.conv.weight.squeeze()

            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W.unsqueeze(0).float()).squeeze()
            z = self.conv(z)
            return z, log_det_W


        def infer(self, z):
            # shape
            batch_size, group_size, n_of_groups = z.size()

            W = self.conv.weight.squeeze()

            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor' or z.type() == 'torch.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/waveglow/model.py:94-166 ---
    class WN(torch.nn.Module):
        """
        This is the WaveNet like layer for the affine coupling.  The primary
        difference from WaveNet is the convolutions need not be causal.  There is
        also no dilation size reset.  The dilation only doubles on each layer
        """

        def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels,
                     kernel_size):
            super(WN, self).__init__()
            assert(kernel_size % 2 == 1)
            assert(n_channels % 2 == 0)
            self.n_layers = n_layers
            self.n_channels = n_channels
            self.in_layers = torch.nn.ModuleList()
            self.res_skip_layers = torch.nn.ModuleList()
            self.cond_layers = torch.nn.ModuleList()

            start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
            start = torch.nn.utils.weight_norm(start, name='weight')
            self.start = start

            # Initializing last layer to 0 makes the affine coupling layers
            # do nothing at first.  This helps with training stability
            end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
            end.weight.data.zero_()
            end.bias.data.zero_()
            self.end = end

            for i in range(n_layers):
                dilation = 2 ** i
                padding = int((kernel_size * dilation - dilation) / 2)
                in_layer = torch.nn.Conv1d(n_channels, 2 * n_channels, kernel_size,
                                           dilation=dilation, padding=padding)
                in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
                self.in_layers.append(in_layer)

                cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels, 1)
                cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
                self.cond_layers.append(cond_layer)

                # last one is not necessary
                if i < n_layers - 1:
                    res_skip_channels = 2 * n_channels
                else:
                    res_skip_channels = n_channels
                res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
                res_skip_layer = torch.nn.utils.weight_norm(
                    res_skip_layer, name='weight')
                self.res_skip_layers.append(res_skip_layer)

        def forward(self, forward_input):
            audio, spect = forward_input
            audio = self.start(audio)

            for i in range(self.n_layers):
                acts = fused_add_tanh_sigmoid_multiply(
                    self.in_layers[i](audio),
                    self.cond_layers[i](spect),
                    torch.IntTensor([self.n_channels]))

                res_skip_acts = self.res_skip_layers[i](acts)
                if i < self.n_layers - 1:
                    audio = res_skip_acts[:, :self.n_channels, :] + audio
                    skip_acts = res_skip_acts[:, self.n_channels:, :]
                else:
                    skip_acts = res_skip_acts

                if i == 0:
                    output = skip_acts
                else:
                    output = skip_acts + output
            return self.end(output)


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/waveglow/model.py:169-335 ---
    class WaveGlow(torch.nn.Module):
        def __init__(self, n_mel_channels, n_flows, n_group, n_early_every,
                     n_early_size, WN_config):
            super(WaveGlow, self).__init__()

            self.upsample = torch.nn.ConvTranspose1d(n_mel_channels,
                                                     n_mel_channels,
                                                     1024, stride=256)
            assert(n_group % 2 == 0)
            self.n_flows = n_flows
            self.n_group = n_group
            self.n_early_every = n_early_every
            self.n_early_size = n_early_size
            self.WN = torch.nn.ModuleList()
            self.convinv = torch.nn.ModuleList()

            n_half = int(n_group / 2)

            # Set up layers with the right sizes based on how many dimensions
            # have been output already
            n_remaining_channels = n_group
            for k in range(n_flows):
                if k % self.n_early_every == 0 and k > 0:
                    n_half = n_half - int(self.n_early_size / 2)
                    n_remaining_channels = n_remaining_channels - self.n_early_size
                self.convinv.append(Invertible1x1Conv(n_remaining_channels))
                self.WN.append(WN(n_half, n_mel_channels * n_group, **WN_config))
            self.n_remaining_channels = n_remaining_channels

        def forward(self, forward_input):
            """
            forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
            forward_input[1] = audio: batch x time
            """
            spect, audio = forward_input

            #  Upsample spectrogram to size of audio
            spect = self.upsample(spect)
            assert(spect.size(2) >= audio.size(1))
            if spect.size(2) > audio.size(1):
                spect = spect[:, :, :audio.size(1)]

            spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
            spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
            spect = spect.permute(0, 2, 1)

            audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
            output_audio = []
            log_s_list = []
            log_det_W_list = []

            for k in range(self.n_flows):
                if k % self.n_early_every == 0 and k > 0:
                    output_audio.append(audio[:, :self.n_early_size, :])
                    audio = audio[:, self.n_early_size:, :]

                audio, log_det_W = self.convinv[k](audio)
                log_det_W_list.append(log_det_W)

                n_half = int(audio.size(1) // 2)
                audio_0 = audio[:, :n_half, :]
                audio_1 = audio[:, n_half:, :]

                output = self.WN[k]((audio_0, spect))
                log_s = output[:, n_half:, :]
                b = output[:, :n_half, :]
                audio_1 = torch.exp(log_s) * audio_1 + b
                log_s_list.append(log_s)

                audio = torch.cat([audio_0, audio_1], 1)

            output_audio.append(audio)
            return torch.cat(output_audio, 1), log_s_list, log_det_W_list

        def infer(self, spect, sigma=1.0):

            spect = self.upsample(spect)
            # trim conv artifacts. maybe pad spec to kernel multiple
            time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
            spect = spect[:, :, :-time_cutoff]

            spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
            spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
            spect = spect.permute(0, 2, 1)

            audio = torch.randn(spect.size(0),
                                self.n_remaining_channels,
                                spect.size(2), device=spect.device).to(spect.dtype)

            audio = torch.autograd.Variable(sigma * audio)

            for k in reversed(range(self.n_flows)):
                n_half = int(audio.size(1) / 2)
                audio_0 = audio[:, :n_half, :]
                audio_1 = audio[:, n_half:, :]

                output = self.WN[k]((audio_0, spect))
                s = output[:, n_half:, :]
                b = output[:, :n_half, :]
                audio_1 = (audio_1 - b) / torch.exp(s)
                audio = torch.cat([audio_0, audio_1], 1)

                audio = self.convinv[k].infer(audio)

                if k % self.n_early_every == 0 and k > 0:
                    z = torch.randn(spect.size(0), self.n_early_size, spect.size(
                        2), device=spect.device).to(spect.dtype)
                    audio = torch.cat((sigma * z, audio), 1)

            audio = audio.permute(
                0, 2, 1).contiguous().view(
                audio.size(0), -1).data
            return audio


        def infer_onnx(self, spect, z, sigma=0.9):

            spect = self.upsample(spect)
            # trim conv artifacts. maybe pad spec to kernel multiple
            time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
            spect = spect[:, :, :-time_cutoff]

            length_spect_group = spect.size(2)//8
            mel_dim = 80
            batch_size = spect.size(0)

            spect = spect.view((batch_size, mel_dim, length_spect_group, self.n_group))
            spect = spect.permute(0, 2, 1, 3)
            spect = spect.contiguous()
            spect = spect.view((batch_size, length_spect_group, self.n_group*mel_dim))
            spect = spect.permute(0, 2, 1)
            spect = spect.contiguous()

            audio = z[:, :self.n_remaining_channels, :]
            z = z[:, self.n_remaining_channels:self.n_group, :]
            audio = sigma*audio

            for k in reversed(range(self.n_flows)):
                n_half = int(audio.size(1) // 2)
                audio_0 = audio[:, :n_half, :]
                audio_1 = audio[:, n_half:(n_half+n_half), :]

                output = self.WN[k]((audio_0, spect))
                s = output[:, n_half:(n_half+n_half), :]
                b = output[:, :n_half, :]
                audio_1 = (audio_1 - b) / torch.exp(s)
                audio = torch.cat([audio_0, audio_1], 1)
                audio = self.convinv[k].infer(audio)

                if k % self.n_early_every == 0 and k > 0:
                    audio = torch.cat((z[:, :self.n_early_size, :], audio), 1)
                    z = z[:, self.n_early_size:self.n_group, :]

            audio = audio.permute(0,2,1).contiguous().view(batch_size, (length_spect_group * self.n_group))

            return audio


        @staticmethod
        def remove_weightnorm(model):
            waveglow = model
            for WN in waveglow.WN:
                WN.start = torch.nn.utils.remove_weight_norm(WN.start)
                WN.in_layers = remove(WN.in_layers)
                WN.cond_layers = remove(WN.cond_layers)
                WN.res_skip_layers = remove(WN.res_skip_layers)
            return waveglow


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/waveglow/model.py:338-343 ---
    def remove(conv_list):
        new_conv_list = torch.nn.ModuleList()
        for old_conv in conv_list:
            old_conv = torch.nn.utils.remove_weight_norm(old_conv)
            new_conv_list.append(old_conv)
        return new_conv_list


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/waveglow/loss_function.py:30-48 ---
    class WaveGlowLoss(torch.nn.Module):
        def __init__(self, sigma=1.0):
            super(WaveGlowLoss, self).__init__()
            self.sigma = sigma

        def forward(self, model_output, clean_audio):
            # clean_audio is unused;
            z, log_s_list, log_det_W_list = model_output
            for i, log_s in enumerate(log_s_list):
                if i == 0:
                    log_s_total = torch.sum(log_s)
                    log_det_W_total = log_det_W_list[i]
                else:
                    log_s_total = log_s_total + torch.sum(log_s)
                    log_det_W_total += log_det_W_list[i]

            loss = torch.sum(
                z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total  # noqa: E501
            return loss / (z.size(0) * z.size(1) * z.size(2))


    # --- licence header: PyTorch/SpeechSynthesis/Tacotron2/waveglow/data_function.py:1-27 ---
    # *****************************************************************************
    #  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
    #
    #  Redistribution and use in source and binary forms, with or without
    #  modification, are permitted provided that the following conditions are met:
    #      * Redistributions of source code must retain the above copyright
    #        notice, this list of conditions and the following disclaimer.
    #      * Redistributions in binary form must reproduce the above copyright
    #        notice, this list of conditions and the following disclaimer in the
    #        documentation and/or other materials provided with the distribution.
    #      * Neither the name of the NVIDIA CORPORATION nor the
    #        names of its contributors may be used to endorse or promote products
    #        derived from this software without specific prior written permission.
    #
    #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    #  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    #  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    #  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
    #  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    #  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    #  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    #  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    #  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    #  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    #
    # *****************************************************************************\


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/waveglow/data_function.py:33-77 ---
    class MelAudioLoader(torch.utils.data.Dataset):
        """
            1) loads audio,text pairs
            2) computes mel-spectrograms from audio files.
        """

        def __init__(self, dataset_path, audiopaths_and_text, args):
            self.audiopaths_and_text = load_filepaths_and_text(dataset_path, audiopaths_and_text)
            self.max_wav_value = args.max_wav_value
            self.sampling_rate = args.sampling_rate
            self.stft = TacotronSTFT(
                args.filter_length, args.hop_length, args.win_length,
                args.n_mel_channels, args.sampling_rate, args.mel_fmin,
                args.mel_fmax)
            self.segment_length = args.segment_length

        def get_mel_audio_pair(self, filename):
            audio, sampling_rate = load_wav_to_torch(filename)

            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))

            # Take segment
            if audio.size(0) >= self.segment_length:
                max_audio_start = audio.size(0) - self.segment_length
                audio_start = torch.randint(0, max_audio_start + 1, size=(1,)).item()
                audio = audio[audio_start:audio_start+self.segment_length]
            else:
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_length - audio.size(0)), 'constant').data

            audio = audio / self.max_wav_value
            audio_norm = audio.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = melspec.squeeze(0)

            return (melspec, audio, len(audio))

        def __getitem__(self, index):
            return self.get_mel_audio_pair(self.audiopaths_and_text[index][0])

        def __len__(self):
            return len(self.audiopaths_and_text)


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/waveglow/data_function.py:80-85 ---
    def batch_to_gpu(batch):
        x, y, len_y = batch
        x = to_gpu(x).float()
        y = to_gpu(y).float()
        len_y = to_gpu(torch.sum(len_y))
        return ((x, y), y, len_y)


    # --- verbatim: PyTorch/SpeechSynthesis/Tacotron2/models.py:57-62 ---
    def init_bn(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if module.affine:
                module.weight.data.uniform_()
        for child in module.children():
            init_bn(child)

    import tarfile

    root = "/data"
    lists = {name: os.path.join(root, name) for name in ("ljs_audio_text_train_subset_625_filelist.txt", "ljs_audio_text_val_filelist.txt")}
    if not os.path.isdir(os.path.join(root, "LJSpeech-1.1")):
        with tarfile.open(os.path.join(root, "LJSpeech-1.1.tar.bz2")) as t:
            t.extractall(root)
    workers = {"multiprocessing_context": "fork"}

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    amp = False  # --amp-run is not an argument of train.py; args.amp stays False

    args = argparse.Namespace(
        max_wav_value=32768.0, sampling_rate=22050, filter_length=1024, hop_length=256,
        win_length=1024, mel_fmin=0.0, mel_fmax=8000.0, n_mel_channels=80,
        segment_length=segment_length)

    model = WaveGlow(n_mel_channels=80, n_flows=12, n_group=8, n_early_every=4, n_early_size=2,
                     WN_config=dict(n_layers=8, kernel_size=3, n_channels=512))
    init_bn(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=0.0)
    criterion = WaveGlowLoss(sigma=1.0)
    criterion.cuda()

    trainset = MelAudioLoader(root, lists["ljs_audio_text_train_subset_625_filelist.txt"], args)
    valset = MelAudioLoader(root, lists["ljs_audio_text_val_filelist.txt"], args)
    train_loader = DataLoader(trainset, num_workers=8, shuffle=True, sampler=None,
                              batch_size=batch_size, pin_memory=False,
                              drop_last=True, **workers)

    val_loss = 0.0
    train_epoch_items_per_sec = 0.0
    num_iters = 0
    model.train()

    for epoch in range(epochs):
        train_epoch_items_per_sec = 0.0
        num_iters = 0

        for i, batch in enumerate(train_loader):
            torch.cuda.synchronize()
            iter_start_time = time.perf_counter()

            model.zero_grad()
            x, y, num_items = batch_to_gpu(batch)

            with torch.cuda.amp.autocast(enabled=amp):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            reduced_loss = loss.item()
            reduced_num_items = num_items.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            num_iters += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 65504.0)
            optimizer.step()

            model.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            iter_stop_time = time.perf_counter()
            iter_time = iter_stop_time - iter_start_time
            train_epoch_items_per_sec += reduced_num_items / iter_time

        # validate(): train.py:273-321
        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(valset, num_workers=1, shuffle=False, sampler=None,
                                    batch_size=batch_size, pin_memory=False,
                                    drop_last=False, **workers)
            val_loss = 0.0
            val_iters = 0
            for i, batch in enumerate(val_loader):
                x, y, num_items = batch_to_gpu(batch)
                with torch.cuda.amp.autocast(enabled=amp):
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                val_loss += loss.item()
                val_iters += 1
            val_loss = val_loss / val_iters
        model.train()

    return {"train_items_per_sec": train_epoch_items_per_sec / num_iters if num_iters > 0 else 0.0,
            "val_loss": val_loss}


FUNC = train_waveglow
