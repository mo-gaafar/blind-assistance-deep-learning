from TTS.api import TTS

# TTS.list_models()  # just to know the model name (to choose the language and voice)


def text_to_audio(text, filepath):
    # Init TTS with the target model name you choose from the list
    # tts = TTS(model_name="tts_models/en/blizzard2013/capacitron-t2-c150_v2",
    #           progress_bar=True, gpu=False)
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC",
              progress_bar=True, gpu=False)

    tts.tts_to_file(text=text, file_path=filepath)  # Run TTS
