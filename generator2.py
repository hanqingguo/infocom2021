########################################################
# new generator for new dataset
########################################################

import scipy.io
import matplotlib.pyplot as plt

import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from utils.audio import Audio
from utils.hparams import HParam


def Add_noise(signal, noise, SNR):
    """
    Args:
        x: useful signal [-1, 1]
        d: noise signal [-1, 1]
        SNR: signal noise ratio as dB, e.g. 0dB, 10dB, 20dB
    Returns:
        mix noise to useful signal
    """
    P_signal = np.sum(abs(signal) ** 2)
    P_d = np.sum(abs(noise) ** 2)
    P_noise = P_signal / (10 ** (SNR / 10))
    scaled_noise = np.sqrt(P_noise / P_d) * noise
    added_noise = signal + scaled_noise
    return added_noise


def scale1(s):
    # normalize to (-1,1)
    normalized = s / np.max(np.abs(s))
    return normalized


def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))


def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def padsilenceafter(wav, samples):
    return np.hstack((wav, np.zeros(samples)))


def padsilencebefore(wav, samples):
    return np.hstack((np.zeros(samples), wav))


def mix_conversation2(hp, args, audio, num, s1_dvec, s1_target, s2, train):
    srate = hp.audio.sample_rate
    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')
    sub_dir_ = os.path.join(dir_, 'conversation')
    os.makedirs(sub_dir_, exist_ok=True)

    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
        'wav files must be mono, not stereo'

    d, _ = librosa.effects.trim(d, top_db=20)
    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)

    # if reference for d-vector is too short, discard it
    if d.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
        return

    # LibriSpeech dataset have many silent interval, so let's vad-merge them
    # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
    if args.vad == 1:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    fix_length = L / 3 # 1 second
    w1_length = fix_length + random.randint(0, fix_length)
    w2_length = L - w1_length
    target_first = random.choice([0,1])
    if target_first:
        w1 = padsilenceafter(w1[:int(w1_length)], int(w2_length))
        w2 = padsilencebefore(w2[:int(w2_length)], int(w1_length))
    else:
        w1 = padsilencebefore(w1[:int(w1_length)], int(w2_length))
        w2 = padsilenceafter(w2[:int(w2_length)], int(w1_length))

    mixed = w1 + w2

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1 / norm, w2 / norm, mixed / norm

    # save vad & normalized wav files
    target_wav_path = formatter(sub_dir_, hp.form.target.wav, num)
    mixed_wav_path = formatter(sub_dir_, hp.form.mixed.wav, num)
    librosa.output.write_wav(target_wav_path, w1, srate)
    librosa.output.write_wav(mixed_wav_path, mixed, srate)

    # save magnitude & phase spectrograms
    target_mag, target_phase = audio.wav2spec(w1)
    mixed_mag, mixed_phase = audio.wav2spec(mixed)

    target_mag_path = formatter(sub_dir_, hp.form.target.mag, num)
    target_phase_path = formatter(sub_dir_, hp.form.target.phase, num)
    mixed_mag_path = formatter(sub_dir_, hp.form.mixed.mag, num)
    mixed_phase_path = formatter(sub_dir_, hp.form.mixed.phase, num)

    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(target_phase), target_phase_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)
    torch.save(torch.from_numpy(mixed_phase), mixed_phase_path)

    # save selected sample as text file. d-vec will be calculated soon
    dvec_text_path = formatter(sub_dir_, hp.form.dvec, num)
    with open(dvec_text_path, 'w') as f:
        f.write(s1_dvec)


def mix_joint2(hp, args, audio, num, s1_dvec, s1_target, s2, SNR, train):
    # Add SNR option here
    srate = hp.audio.sample_rate
    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')
    sub_dir_ = os.path.join(dir_, 'joint', str(SNR)+'dB')
    os.makedirs(sub_dir_, exist_ok=True)

    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
        'wav files must be mono, not stereo'

    d, _ = librosa.effects.trim(d, top_db=20)
    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)

    # if reference for d-vector is too short, discard it
    if d.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
        return

    # LibriSpeech dataset have many silent interval, so let's vad-merge them
    # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
    if args.vad == 1:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1, w2 = w1[:L], w2[:L]

    mixed = Add_noise(w1, w2, SNR)

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1 / norm, w2 / norm, mixed / norm

    # save vad & normalized wav files
    target_wav_path = formatter(sub_dir_, hp.form.target.wav, num)
    mixed_wav_path = formatter(sub_dir_, hp.form.mixed.wav, num)
    librosa.output.write_wav(target_wav_path, w1, srate)
    librosa.output.write_wav(mixed_wav_path, mixed, srate)

    # save magnitude & phase spectrograms
    target_mag, target_phase = audio.wav2spec(w1)
    mixed_mag, mixed_phase = audio.wav2spec(mixed)

    target_mag_path = formatter(sub_dir_, hp.form.target.mag, num)
    target_phase_path = formatter(sub_dir_, hp.form.target.phase, num)
    mixed_mag_path = formatter(sub_dir_, hp.form.mixed.mag, num)
    mixed_phase_path = formatter(sub_dir_, hp.form.mixed.phase, num)

    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(target_phase), target_phase_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)
    torch.save(torch.from_numpy(mixed_phase), mixed_phase_path)

    # save selected sample as text file. d-vec will be calculated soon
    dvec_text_path = formatter(sub_dir_, hp.form.dvec, num)
    with open(dvec_text_path, 'w') as f:
        f.write(s1_dvec)


def mix_noise(hp, args, audio, num, s1_dvec, s1_target, noise_mat, noise_type, SNR, train):
    srate = hp.audio.sample_rate

    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')
    sub_dir_ = os.path.join(dir_, 'noise', noise_type, str(SNR)+'dB')
    os.makedirs(sub_dir_, exist_ok=True)
    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    mat = scipy.io.loadmat(noise_mat)
    noise = scale1(np.squeeze(mat[noise_type]))

    assert len(d.shape) == len(w1.shape) == len(noise.shape) == 1, \
        'wav files must be mono, not stereo'

    d, _ = librosa.effects.trim(d, top_db=20)
    w1, _ = librosa.effects.trim(w1, top_db=20)

    # if reference for d-vector is too short, discard it
    if d.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
        return

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or noise.shape[0] < L:
        return
    w1, noise = w1[:L], noise[:L]
    mixed = Add_noise(signal=w1, noise=noise, SNR=SNR)

    norm = np.max(np.abs(mixed)) * 1.1
    w1, mixed = w1 / norm, mixed / norm

    # save vad & normalized wav files
    target_wav_path = formatter(sub_dir_, hp.form.target.wav, num)
    mixed_wav_path = formatter(sub_dir_, hp.form.mixed.wav, num)
    librosa.output.write_wav(target_wav_path, w1, srate)
    librosa.output.write_wav(mixed_wav_path, mixed, srate)

    # save magnitude & phase spectrograms
    target_mag, target_phase = audio.wav2spec(w1)
    mixed_mag, mixed_phase = audio.wav2spec(mixed)

    target_mag_path = formatter(sub_dir_, hp.form.target.mag, num)
    target_phase_path = formatter(sub_dir_, hp.form.target.phase, num)
    mixed_mag_path = formatter(sub_dir_, hp.form.mixed.mag, num)
    mixed_phase_path = formatter(sub_dir_, hp.form.mixed.phase, num)

    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(target_phase), target_phase_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)
    torch.save(torch.from_numpy(mixed_phase), mixed_phase_path)

    # save selected sample as text file. d-vec will be calculated soon
    dvec_text_path = formatter(sub_dir_, hp.form.dvec, num)
    with open(dvec_text_path, 'w') as f:
        f.write(s1_dvec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-d', '--libri_dir', type=str, default=None,
                        help="Directory of LibriSpeech dataset, containing folders of train-clean-100, train-clean-360, dev-clean.")
    parser.add_argument('-v', '--voxceleb_dir', type=str, default=None,
                        help="Directory of VoxCeleb2 dataset, ends with 'aac'")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-p', '--process_num', type=int, default=None,
                        help='number of processes to run. default: cpu_count')
    parser.add_argument('--vad', type=int, default=0,
                        help='apply vad to wav file. yes(1) or no(0, default)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)

    hp = HParam(args.config)

    cpu_num = cpu_count() if args.process_num is None else args.process_num

    if args.libri_dir is None and args.voxceleb_dir is None:
        raise Exception("Please provide directory of data")

    if args.libri_dir is not None:
        train_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-360', '*'))
                         if os.path.isdir(x)]
        # [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-100', '*'))
        #                     if os.path.isdir(x)] + \
        #                 [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-360', '*'))
        #                     if os.path.isdir(x)]
        # we recommned to exclude train-other-500
        # See https://github.com/mindslab-ai/voicefilter/issues/5#issuecomment-497746793
        # + \
        # [x for x in glob.glob(os.path.join(args.libri_dir, 'train-other-500', '*'))
        #    if os.path.isdir(x)]
        test_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'dev-clean', '*'))]

    elif args.voxceleb_dir is not None:
        all_folders = [x for x in glob.glob(os.path.join(args.voxceleb_dir, '*'))
                       if os.path.isdir(x)]
        train_folders = all_folders[:-20]
        test_folders = all_folders[-20:]

    train_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                 for spk in train_folders]
    train_spk = [x for x in train_spk if len(x) >= 2]

    test_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                for spk in test_folders]

    test_spk = [x for x in test_spk if len(x) >= 2]  # list of list, each speaker has a list of wavs

    selected_spk = ['84', '174', '422']
    specific_spk = []
    rest_spk = []
    for spk in test_spk:
        if spk[0].split('/')[-3] not in selected_spk:
            rest_spk.append(spk)
        else:
            specific_spk.append(spk)

    audio = Audio(hp)


    def train_wrapper(num):
        spk1, spk2 = random.sample(train_spk, 2)
        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)
        SNR = random.choice([0, 10, 20])
        mix_joint2(hp, args, audio, num, s1_dvec, s1_target, s2, SNR=SNR, train=True)
        mix_conversation2(hp, args, audio, num, s1_dvec, s1_target, s2, train=True)
        noise_source = glob.glob('./noise_source/*')
        noise_mat = random.choice(noise_source)
        noise_type = noise_mat.split('/')[-1][:-4]

        mix_noise(hp, args, audio, num, s1_dvec, s1_target, noise_mat, noise_type, SNR, train=True)


    def test_wrapper(num):
        spk1 = random.sample(specific_spk, 1)[0]
        spk2 = random.sample(rest_spk, 1)[0]
        # spk1, spk2 = random.sample(test_spk, 2)

        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)
        SNR = random.choice([0, 10, 20])
        mix_joint2(hp, args, audio, num, s1_dvec, s1_target, s2, SNR=SNR, train=False)
        mix_conversation2(hp, args, audio, num, s1_dvec, s1_target, s2, train=False)
        noise_source = glob.glob('./noise_source/*')
        noise_mat = random.choice(noise_source)
        noise_type = noise_mat.split('/')[-1][:-4]
        mix_noise(hp, args, audio, num, s1_dvec, s1_target, noise_mat, noise_type, SNR, train=False)
    arr = list(range(10 ** 2))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(train_wrapper, arr), total=len(arr)))

    # arr = list(range(10 ** 2))
    arr = list(range(10))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(test_wrapper, arr), total=len(arr)))
