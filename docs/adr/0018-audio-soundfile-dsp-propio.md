# ADR-0018: Audio con soundfile y DSP propio (sin torchaudio)
- Estado: aceptado
- Fecha: 2026-09-27
- Contexto: SPEC §5.1 sugiere torchaudio para audio. Desde 2025 torchaudio está en modo mantenimiento y su IO pasó a otra librería; su publicación sincronizada con cada versión de torch (2.14 en el lock) no está garantizada, y una rueda desalineada rompería la instalación por variante de hardware (RF-TRN-02).
- Decisión: IO con `soundfile` (BSD, trae libsndfile para Windows y Linux: wav, flac, ogg, mp3). Resample con `scipy.signal.resample_poly` (scipy ya es dependencia de scikit-learn). Mel-spectrogram, MFCC (DCT-II del log-mel) y SpecAugment implementados sobre `torch.stft` en `perceptron.data.audio`. Los espectrogramas entran a los mismos backbones de visión (`vision.small_cnn`, `vision.timm_backbone`) como imágenes de 1 canal, igual que el ejemplo de §9.2.
- Consecuencias: una dependencia menos atada a la versión de torch; el DSP es chico, determinístico y testeado (pico de energía en la banda correcta). Si hiciera falta un front-end más completo (p. ej. resample de alta calidad o codecs raros), se evalúa una dependencia adicional.
- Alternativas consideradas: torchaudio (riesgo de versiones); librosa (ISC, pero arrastra numba/llvmlite, pesado para el runtime embebido).
