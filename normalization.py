"""Четыре варианта предобработки интенсивности МРТ (DWI, ADC, FLAIR), сравниваемые в работе.

Вариант 1 (baseline) воспроизводит проект: MONAI NormalizeIntensityd(nonzero=True, channel_wise=True).
Варианты 2-4 добавлены в рамках исследования.
"""
import numpy as np
from scipy import ndimage

CLIP_PERCENTILES = (0.5, 99.5)
ADC_MAX = 3.0  # 10^-3 мм²/с: ADC свободной воды при 37 °C около 3·10^-3 мм²/с
EPS = 1e-8


def brain_mask_from_adc(adc):
    """Общая маска мозга: наибольшая связная область ADC != 0 с заполненными полостями.

    Данные ISLES 2022 по большей части очищены от черепа, но DWI отдельных случаев
    содержит ненулевой фон (например, sub-strokecase0046), поэтому маска строится по ADC.
    """
    support = np.asarray(adc) != 0
    labels, count = ndimage.label(support, structure=np.ones((3, 3, 3)))
    if count == 0:
        raise ValueError("ADC не содержит ненулевых вокселей")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = labels == sizes.argmax()
    return ndimage.binary_fill_holes(largest)


def z_normalize_nonzero(volume):
    """Вариант 1: Z-нормализация по ненулевым вокселям"""
    volume = np.asarray(volume, dtype=np.float32)
    nonzero = volume != 0
    if not nonzero.any():
        raise ValueError("Все воксели равны нулю")
    values = volume[nonzero]
    std = values.std()
    if std == 0:
        std = 1.0
    out = np.zeros_like(volume, dtype=np.float32)
    out[nonzero] = (values - values.mean()) / std
    return out


def z_normalize(volume, brain_mask, clip_percentiles=None, eps=EPS):
    """Z-нормализация внутри маски мозга, при необходимости после ограничения перцентилями."""
    volume = np.asarray(volume, dtype=np.float32)
    mask = np.asarray(brain_mask, dtype=bool)
    if volume.shape != mask.shape:
        raise ValueError("Размеры изображения и маски не совпадают")
    if not np.isfinite(volume).all():
        raise ValueError("В изображении присутствуют NaN или Inf")
    if not mask.any():
        raise ValueError("Маска мозга пуста")
    values = volume[mask]
    if clip_percentiles is not None:
        low_p, high_p = clip_percentiles
        if not (0 <= low_p < high_p <= 100):
            raise ValueError("Некорректные перцентили")
        low, high = np.percentile(values, [low_p, high_p])
        values = np.clip(values, low, high)
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (values - values.mean()) / (values.std() + eps)
    return out


def adc_to_common_units(adc, brain_mask):
    """Привести ADC к единицам 10^-3 мм²/с.

    В ISLES 2022 карты ADC хранятся в трёх разных единицах: мм²/с (медиана по мозгу около 0,0009),
    10^-3 мм²/с (около 0,9) и 10^-6 мм²/с (около 900). Единицы определяются по медиане внутри маски.
    """
    adc = np.asarray(adc, dtype=np.float32)
    median = float(np.median(adc[brain_mask]))
    if median < 0.05:
        factor = 1000.0
    elif median >= 50.0:
        factor = 1.0 / 1000.0
    else:
        factor = 1.0
    return adc * factor


def scale_adc(adc, brain_mask, adc_max=ADC_MAX):
    adc = adc_to_common_units(adc, brain_mask)
    out = np.zeros_like(adc, dtype=np.float32)
    out[brain_mask] = np.clip(adc[brain_mask], 0.0, adc_max) / adc_max
    return out


def preprocess(image, brain_mask, variant):
    """image: массив [3, X, Y, Z] с каналами DWI, ADC, FLAIR; возвращает массив той же формы."""
    dwi, adc, flair = (np.asarray(c, dtype=np.float32) for c in image)
    if variant == 1:
        channels = [z_normalize_nonzero(c) for c in (dwi, adc, flair)]
    elif variant == 2:
        channels = [z_normalize(c, brain_mask) for c in (dwi, adc, flair)]
    elif variant == 3:
        channels = [z_normalize(c, brain_mask, CLIP_PERCENTILES) for c in (dwi, adc, flair)]
    elif variant == 4:
        channels = [z_normalize(dwi, brain_mask, CLIP_PERCENTILES),
                    scale_adc(adc, brain_mask),
                    z_normalize(flair, brain_mask, CLIP_PERCENTILES)]
    else:
        raise ValueError(f"Неизвестный вариант: {variant}")
    return np.stack(channels, axis=0).astype(np.float32)
