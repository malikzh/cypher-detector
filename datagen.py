#!/usr/bin/env python3
from loguru import logger as log
from encoder import ENCODER_FACTORY
from os import mkdir, urandom
from os.path import join, abspath, isdir

DATASET_DIR = '_dataset'
QUANTITY = 4096  # Количество шифротекста на каждый класс
TEXT_SIZE = 1024  # Размер текста в байтах
UPDATE_IV_AND_KEY_EVERY = 256  # Каждые n раз обновляем ключ и IV

log.info("Data generation started.")

if not isdir(DATASET_DIR):
    mkdir(DATASET_DIR)

# Генерация текстов
TEXTS = list([i.to_bytes(TEXT_SIZE, byteorder="big") for i in range(QUANTITY)])

for enc_name, enc_factory in ENCODER_FACTORY.items():
    log.info(f"Generating data for {enc_name}...")

    dir_path = join(DATASET_DIR, enc_name)

    if not isdir(dir_path):
        mkdir(dir_path)

    # Generate texts
    encoder = enc_factory()

    for i in range(QUANTITY):
        log.info(f"  [{enc_name}] Generating {i + 1}/{QUANTITY} item...")
        text = TEXTS[i]

        if i % UPDATE_IV_AND_KEY_EVERY == 0:
            log.info(f"    [{enc_name}] Updating key and IV...")
            encoder.generate()

        ciphertext = encoder.encrypt(text)

        # Сохраняем зашифрованные данные в файл
        with open(join(dir_path, f"{i + 1}.bin"), "wb") as f:
            f.write(ciphertext)

log.info("Data generation completed.")
