import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources


def tensor_normalize(S):
    temp_max, _ = torch.max(S, dim=1)
    batch_max, _ = torch.max(temp_max, dim=1)
    batch_max = torch.reshape(batch_max, (1, 1, 1))
    temp_min, _ = torch.min(S, dim=1)
    batch_min, _ = torch.min(temp_min, dim=1)
    batch_min = torch.reshape(batch_min, (1, 1, 1))
    normalized_S = (S - batch_min) / (batch_max - batch_min)
    return normalized_S


def validate(audio, model, embedder, testloader, writer, step):
    model.eval()

    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in testloader:
            dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase = batch[0]

            dvec_mel = dvec_mel.cuda()
            target_mag = target_mag.unsqueeze(0).cuda()
            mixed_mag = mixed_mag.unsqueeze(0).cuda()

            dvec = embedder(dvec_mel)
            dvec = dvec.unsqueeze(0)
            est_noise_mag = model(mixed_mag, dvec)
            # est_noise_mag.size() = [1, 301, 601]
            est_purified_mag = tensor_normalize(mixed_mag - est_noise_mag)
            test_loss = criterion(target_mag, est_purified_mag).item()

            mixed_mag = mixed_mag[0].cpu().detach().numpy()
            target_mag = target_mag[0].cpu().detach().numpy()
            est_purified_mag = est_purified_mag[0].cpu().detach().numpy()
            est_noise_mag = est_noise_mag[0].cpu().detach().numpy()
            est_noise_wav = audio.spec2wav(est_noise_mag, mixed_phase)
            est_purified_wav = mixed_wav - est_noise_wav
            sdr = bss_eval_sources(target_wav, est_purified_wav, False)[0][0]
            writer.log_evaluation(test_loss, sdr,
                                  mixed_wav, est_noise_wav, est_purified_wav, target_wav,
                                  mixed_mag.T, target_mag.T, est_purified_mag.T, est_noise_mag.T,
                                  step)
            break

    model.train()
