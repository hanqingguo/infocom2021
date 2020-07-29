# Use dataloader to load multiple data, then use inference.py to produce output.
# combine with speech2text.py to produce metrics

from datasets.eva_dataloader import create_dataloader
import wavio
import argparse
from utils.hparams import HParam
import os
import glob
import torch
import librosa
import argparse

from utils.audio import Audio
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
from utils.speech2text import speech_2_text
from utils.evaluation import tensor_normalize
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def wav_to_embedding(wav, embedder):
    """
    This function takes wav as input, then produce the embedding for given wav
    Returns:
    embedding as numpy array
    """
    wav_mel = audio.get_mel(wav)
    wav_dvec_mel = torch.from_numpy(wav_mel).float().cuda()
    wav_dvec = embedder(wav_dvec_mel)
    wav_dvector = wav_dvec.unsqueeze(0)
    wav_dvector = wav_dvector.cpu().detach().numpy()

    return wav_dvector

def init_badwav():
    wer = 1
    mer = 1
    wil = 1
    pesq_value = 0
    sdr = [-10]
    sir = [np.inf]
    sar = [-10]
    return wer, mer, wil, pesq_value, sdr, sir, sar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Root directory of run.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model. Used for both logging and saving checkpoints.")
    parser.add_argument('-g', '--gpu', type=int, required=True, default='1',
                        help="ID of the selected gpu. Used for gpu selection.")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="out directory of result.wav")
    parser.add_argument('-x', '--xlsx', type=str, required=True,
                        help="result name of xlsx file")
    args = parser.parse_args()

    hp = HParam(args.config)
    testloader = create_dataloader(hp, args, train=False)
    result_purified1 = []
    result_purified2 = []
    result_purified3 = []
    result_mixed = []
    for batch in testloader:
        # length of batch is 1, set in dataloader
        dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase, dvec_path, target_wav_path, mixed_wav_path = \
            batch[0]
        print("Target: {}".format(target_wav_path))
        print("Mixed: {}".format(mixed_wav_path))
        model = VoiceFilter(hp).cuda()
        chkpt_model = torch.load(args.checkpoint_path)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        embedder = SpeechEmbedder(hp).cuda()
        chkpt_embed = torch.load(args.embedder_path)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()

        audio = Audio(hp)
        dvec_wav, _ = librosa.load(dvec_path, sr=16000)
        dvec_mel = audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().cuda()
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)  # (1, 256)

        mixed_wav, _ = librosa.load(mixed_wav_path, sr=16000)
        mixed_mag, mixed_phase = audio.wav2spec(mixed_wav)
        mixed_mag = torch.from_numpy(mixed_mag).float().cuda()

        mixed_mag = mixed_mag.unsqueeze(0)

        est_noise_mag = model(mixed_mag, dvec)
        # est_noise_mag.size() = [1, 301, 601]
        est_purified_mag = tensor_normalize(mixed_mag + est_noise_mag)
        est_purified_mag = est_purified_mag[0].cpu().detach().numpy()
        mixed_mag = mixed_mag[0].cpu().detach().numpy()
        target_mag = target_mag[0].cpu().detach().numpy()
        # est_purified_mag = est_purified_mag[0].cpu().detach().numpy()
        est_noise_mag = est_noise_mag[0].cpu().detach().numpy()
        est_noise_wav = audio.spec2wav(est_noise_mag, mixed_phase)

        # scale is frequency pass to time domain, used on wav signal normalization
        est_purified_wav1 = audio.spec2wav(est_purified_mag, mixed_phase)  # path 1
        enhanced_wav = (mixed_wav + 100 * est_noise_wav) / max(abs(mixed_wav + 100 * est_noise_wav))  # path 2

        os.makedirs(args.out_dir, exist_ok=True)
        purified1 = os.path.join(args.out_dir, 'result1.wav')
        purified2 = os.path.join(args.out_dir, 'result2.wav')
        purified3 = os.path.join(args.out_dir, 'result3.wav')
        target_path = os.path.join(args.out_dir, 'target.wav')
        mixed_path = os.path.join(args.out_dir, 'mixed.wav')
        # original mixed wav and target wav are not PCM, cannot be read by google cloud
        wavio.write(purified1, est_purified_wav1, 16000, sampwidth=2)  # frequency +
        wavio.write(purified2, est_noise_wav, 16000, sampwidth=2)  # est noise
        wavio.write(purified3, enhanced_wav, 16000, sampwidth=2)  # mix + est noise
        wavio.write(target_path, target_wav, 16000, sampwidth=2)
        wavio.write(mixed_path, mixed_wav, 16000, sampwidth=2)
        # librosa.output.write_wav(out_path, enhanced_wav, sr=16000)
        dvec = dvec.cpu().detach().numpy()  # the reference embedding
        mixed_dvec = wav_to_embedding(mixed_wav, embedder)

        purified1_dvec = wav_to_embedding(est_purified_wav1, embedder)
        purified2_dvec = wav_to_embedding(est_noise_wav, embedder)
        purified3_dvec = wav_to_embedding(enhanced_wav, embedder)

        purified1_conf = cosine_similarity(dvec, purified1_dvec)[0][0]
        purified2_conf = cosine_similarity(dvec, purified2_dvec)[0][0]
        purified3_conf = cosine_similarity(dvec, purified3_dvec)[0][0]
        mixed_conf = cosine_similarity(dvec, mixed_dvec)[0][0]

        try:
            [wer, mer, wil], pesq_value, sdr, sir, sar = speech_2_text(target_path, purified1, 'google')
        except TypeError:
            wer, mer, wil, pesq_value, sdr, sir, sar = init_badwav()
            # print(
            #     "Spectrogram: wer: {0}, mer: {1}, wil: {2}, pesq: {3}, sdr: {4}".format(wer, mer, wil, pesq_value, sdr))
        r1 = {"mixed_path": mixed_wav_path, "target_path": target_wav_path,
              "wer": wer, "mer": mer, "wil": wil, "pesq": pesq_value, "sdr": sdr[0], "sir": sir[0], "sar": sar[0],
              "confidence": purified1_conf}
        result_purified1.append(r1)

        try:
            [wer, mer, wil], pesq_value, sdr, sir, sar = speech_2_text(target_path, purified2, 'google')
        except TypeError:
            wer, mer, wil, pesq_value, sdr, sir, sar = init_badwav()

        r2 = {"mixed_path": mixed_wav_path, "target_path": target_wav_path,
              "wer": wer, "mer": mer, "wil": wil, "pesq": pesq_value, "sdr": sdr[0], "sir": sir[0], "sar": sar[0],
              "confidence": purified2_conf}
        result_purified2.append(r2)

        try:
            [wer, mer, wil], pesq_value, sdr, sir, sar = speech_2_text(target_path, purified3, 'google')
        except TypeError:
            wer, mer, wil, pesq_value, sdr, sir, sar = init_badwav()
        r3 = {"mixed_path": mixed_wav_path, "target_path": target_wav_path,
              "wer": wer, "mer": mer, "wil": wil, "pesq": pesq_value, "sdr": sdr[0], "sir": sir[0], "sar": sar[0],
              "confidence": purified3_conf}
        result_purified3.append(r3)

        try:
            [wer, mer, wil], pesq_value, sdr, sir, sar = speech_2_text(target_path, mixed_path, 'google')
        except TypeError:
            wer, mer, wil, pesq_value, sdr, sir, sar = init_badwav()
        r4 = {"mixed_path": mixed_wav_path, "target_path": target_wav_path,
              "wer": wer, "mer": mer, "wil": wil, "pesq": pesq_value, "sdr": sdr[0], "sir": sir[0], "sar": sar[0],
              "confidence": mixed_conf}
        result_mixed.append(r4)

    writer = pd.ExcelWriter(args.xlsx, engine='xlsxwriter', options={'strings_to_urls': False})
    df1 = pd.DataFrame(data=result_purified1)
    df2 = pd.DataFrame(data=result_purified2)
    df3 = pd.DataFrame(data=result_purified3)
    df4 = pd.DataFrame(data=result_mixed)
    df1.to_excel(writer, sheet_name='purified1')
    df2.to_excel(writer, sheet_name='purified2')
    df3.to_excel(writer, sheet_name='purified3')
    df4.to_excel(writer, sheet_name='mixed')
    writer.close()
