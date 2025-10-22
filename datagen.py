#!/usr/bin/env python3
from  loguru import logger as log
from encoder import ENCODER_FACTORY
from os import mkdir, urandom
from os.path import join, abspath, isdir

DATASET_DIR = '_dataset'
QUANTITY = 2000 # Количество шифротекста на каждый класс
TEXT_SIZE = 1024 # Размер текста в байтах

log.info("Data generation started.")


if not isdir(DATASET_DIR):
    mkdir(DATASET_DIR)

# Генерация текстов
TEXTS = []
for i in range(QUANTITY):
    TEXTS.append(i.to_bytes(TEXT_SIZE, byteorder="big"))

# Генерация ключей
KEYS = dict({enc_name:enc_factory().generate_key() for enc_name, enc_factory in ENCODER_FACTORY.items()})

for enc_name, enc_factory in ENCODER_FACTORY.items():
    log.info(f"Generating data for {enc_name}...")

    dir_path = join(DATASET_DIR, enc_name)

    if not isdir(dir_path):
        mkdir(dir_path)

    # Generate texts
    encoder = enc_factory()

    for i in range(QUANTITY):
        log.info(f"  Generating {i+1}/{QUANTITY} item...")

        key = KEYS[enc_name]
        text = TEXTS[i]

        ciphertext = encoder.encrypt(text, key)

        # Сохраняем зашифрованные данные в файл
        with open(join(dir_path, f"{i+1}.bin"), "wb") as f:
            f.write(ciphertext)

log.info("Data generation completed.")