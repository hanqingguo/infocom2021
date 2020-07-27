import speech_recognition as sr
from os import path
import jiwer
from pesq import pesq
from scipy.io import wavfile
import librosa
import mir_eval


def metric_for_text(text_groundtruth, text_estimated):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemoveWhiteSpace(replace_by_space=False),
        jiwer.SentencesToListOfWords(word_delimiter=" ")
    ])
    measures = jiwer.compute_measures(text_groundtruth, text_estimated, truth_transform=transformation,
                                      hypothesis_transform=transformation)
    measures = jiwer.compute_measures(text_groundtruth, text_estimated)
    wer = measures['wer']
    mer = measures['mer']
    wil = measures['wil']

    return [wer, mer, wil]


def speech_2_text(wav_groundtruth=path.join(path.dirname(path.realpath(__file__)), "speakerA.wav"),
                  wav_estimated=path.join(path.dirname(path.realpath(__file__)), "speakerB.wav"), engine_name='google'):
    r = sr.Recognizer()
    with sr.AudioFile(wav_groundtruth) as source:
        audio = r.record(source)

    with sr.AudioFile(wav_estimated) as source:
        audio_estimated = r.record(source)

    rate, audio_pesq = wavfile.read(wav_groundtruth)
    rate, audio_estimated_pesq = wavfile.read(wav_estimated)

    (sdr, sir, sar, _) = mir_eval.separation.bss_eval_sources(audio_pesq, audio_estimated_pesq, compute_permutation=True)

    if engine_name == 'google':
        GOOGLE_CLOUD_SPEECH_CREDENTIALS_PATH = path.join(path.dirname(path.realpath(__file__)), "key.json")
        with open(GOOGLE_CLOUD_SPEECH_CREDENTIALS_PATH, 'r') as file:
            GOOGLE_CLOUD_SPEECH_CREDENTIALS = file.read()
        try:
            text_groundtruth = r.recognize_google_cloud(audio, language="en-us",
                                                        credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
            text_estimated = r.recognize_google_cloud(audio_estimated, language="en-us",
                                                      credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
            print("Groundtruth:{0}; Estimated:{1}".format(text_groundtruth, text_estimated))
            print(pesq(rate, audio_pesq, audio_estimated_pesq, 'wb'))
            # return the results
            return metric_for_text(text_groundtruth, text_estimated), pesq(rate, audio_pesq, audio_estimated_pesq, 'wb'), sdr, sir, sar

        except sr.UnknownValueError:
            print("Google Cloud Speech could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Cloud Speech service; {0}".format(e))
    else:
        # recognize speech using Sphinx
        try:
            print("Sphinx thinks you said " + r.recognize_sphinx(audio))
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))

        # recognize speech using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            print("Google Speech Recognition thinks you said " + r.recognize_google(audio))
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        # recognize speech using Wit.ai
        WIT_AI_KEY = "INSERT WIT.AI API KEY HERE"  # Wit.ai keys are 32-character uppercase alphanumeric strings
        try:
            print("Wit.ai thinks you said " + r.recognize_wit(audio, key=WIT_AI_KEY))
        except sr.UnknownValueError:
            print("Wit.ai could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Wit.ai service; {0}".format(e))

        # recognize speech using Microsoft Azure Speech
        AZURE_SPEECH_KEY = "INSERT AZURE SPEECH API KEY HERE"  # Microsoft Speech API keys 32-character lowercase hexadecimal strings
        try:
            print("Microsoft Azure Speech thinks you said " + r.recognize_azure(audio, key=AZURE_SPEECH_KEY))
        except sr.UnknownValueError:
            print("Microsoft Azure Speech could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Microsoft Azure Speech service; {0}".format(e))

        # recognize speech using Microsoft Bing Voice Recognition
        BING_KEY = "INSERT BING API KEY HERE"  # Microsoft Bing Voice Recognition API keys 32-character lowercase hexadecimal strings
        try:
            print("Microsoft Bing Voice Recognition thinks you said " + r.recognize_bing(audio, key=BING_KEY))
        except sr.UnknownValueError:
            print("Microsoft Bing Voice Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))

        # recognize speech using Houndify
        HOUNDIFY_CLIENT_ID = "INSERT HOUNDIFY CLIENT ID HERE"  # Houndify client IDs are Base64-encoded strings
        HOUNDIFY_CLIENT_KEY = "INSERT HOUNDIFY CLIENT KEY HERE"  # Houndify client keys are Base64-encoded strings
        try:
            print("Houndify thinks you said " + r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID,
                                                                     client_key=HOUNDIFY_CLIENT_KEY))
        except sr.UnknownValueError:
            print("Houndify could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Houndify service; {0}".format(e))

        # recognize speech using IBM Speech to Text
        IBM_USERNAME = "INSERT IBM SPEECH TO TEXT USERNAME HERE"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
        IBM_PASSWORD = "INSERT IBM SPEECH TO TEXT PASSWORD HERE"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
        try:
            print("IBM Speech to Text thinks you said " + r.recognize_ibm(audio, username=IBM_USERNAME,
                                                                          password=IBM_PASSWORD))
        except sr.UnknownValueError:
            print("IBM Speech to Text could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from IBM Speech to Text service; {0}".format(e))

if __name__ == '__main__':

    wav_groundtruth = path.join(path.dirname(path.realpath(__file__)), "000030-target.wav")
    wav_estimated = path.join(path.dirname(path.realpath(__file__)), "000030-target.wav")
    # wav_groundtruth = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
    # wav_groundtruth = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")
    # sr, wav = wavfile.read(wav_groundtruth)
    # wav2, _ = librosa.load(wav_groundtruth, sr=16000)
    # print(wav)
    # print(wav2)
    # print(wav2*255)
    # use the audio file as the audio source
    engine_name = 'google'
    [wer, mer, wil], pesq_value = speech_2_text(wav_groundtruth, wav_estimated, engine_name)
    print("wer: {0}, mer: {1}, wil: {2}, pesq: {3}".format(wer, mer, wil, pesq_value))
