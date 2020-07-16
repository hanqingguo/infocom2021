import numpy as np
from tensorboardX import SummaryWriter

from .plotting import plot_spectrogram_to_numpy


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation(self, test_loss, sdr,
                       mixed_wav, est_noise_wav, est_purified_wav1, est_purified_wav2, est_purified_wav3, target_wav,
                       new_target_wav,
                       mixed_spec, target_spec, new_target_spec, est_purified_mag, est_noise_mag,
                       step):
        
        self.add_scalar('test_loss', test_loss, step)
        self.add_scalar('SDR', sdr, step)

        self.add_audio('mixed_wav', mixed_wav, step, self.hp.audio.sample_rate)
        self.add_audio('target_wav', target_wav, step, self.hp.audio.sample_rate)
        self.add_audio('new_target_wav', new_target_wav, step, self.hp.audio.sample_rate)
        self.add_audio('est_noise_wav', est_noise_wav, step, self.hp.audio.sample_rate)
        self.add_audio('est_purified_wav1', est_purified_wav1, step, self.hp.audio.sample_rate )
        self.add_audio('est_purified_wav2', est_purified_wav2, step, self.hp.audio.sample_rate)
        self.add_audio('est_purified_wav3', est_purified_wav3, step, self.hp.audio.sample_rate)

        self.add_image('data/mixed_spectrogram',
            plot_spectrogram_to_numpy(mixed_spec), step, dataformats='HWC')
        self.add_image('result/est_noise_spectrogram',
            plot_spectrogram_to_numpy(est_noise_mag), step, dataformats='HWC')
        self.add_image('result/est_purified_spectrogram',
            plot_spectrogram_to_numpy(est_purified_mag), step, dataformats='HWC')
        self.add_image('data/target_spectrogram',
            plot_spectrogram_to_numpy(target_spec), step, dataformats='HWC')
        self.add_image('data/new_target_spectrogram',
            plot_spectrogram_to_numpy(new_target_spec), step, dataformats='HWC')
        self.add_image('result/estimation_error_sq',
            plot_spectrogram_to_numpy(np.square(est_purified_mag - new_target_spec)), step, dataformats='HWC')
