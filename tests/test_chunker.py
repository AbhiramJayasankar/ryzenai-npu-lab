"""CPU-only streaming checks with deterministic synthetic 16 kHz audio."""

import numpy as np
import pytest

from apps.npu_dictate.chunker import (
    BLOCK,
    LIMIT,
    MAX_BLOCKS,
    MIN_VOICED,
    PAD,
    PAUSE_BLOCKS,
    SR,
    Chunker,
    rms,
)


def silence(samples):
    return np.zeros(samples, dtype=np.float32)


def noise(samples, level=0.04):
    return np.random.default_rng(42).normal(0, level, samples).astype(np.float32)


def collect(audio, chunk_size):
    chunker = Chunker()
    pieces = []
    for start in range(0, len(audio), chunk_size):
        pieces.extend(chunker.push(audio[start:start + chunk_size]))
    pieces.extend(chunker.finish())
    return pieces


@pytest.mark.parametrize("samples", [0, SR // 10 + 17, 25 * SR])
def test_silence_produces_no_pieces(samples):
    assert collect(silence(samples), 1024) == []


def test_low_rms_background_noise_produces_no_pieces():
    assert collect(noise(12 * SR, level=0.0005), 1024) == []


def test_short_utterance_followed_by_silence():
    # Start quietly so the first block establishes a background noise floor.
    audio = np.concatenate([silence(10 * BLOCK), noise(30 * BLOCK), silence(30 * BLOCK)])
    chunker = Chunker()
    # Automatic pause cuts only start once six seconds have accumulated.
    assert chunker.push(audio) == []
    pieces = chunker.finish()
    assert len(pieces) == 1
    np.testing.assert_array_equal(pieces[0], np.concatenate([audio, silence(PAD)]))
    assert len(pieces[0]) <= LIMIT


def test_pause_emits_a_piece_while_streaming():
    audio = np.concatenate([
        silence(10 * BLOCK), noise(200 * BLOCK), silence(PAUSE_BLOCKS * BLOCK),
    ])
    chunker = Chunker()
    pieces = chunker.push(audio)
    assert len(pieces) == 1
    # Half the detected pause stays buffered; the emitted half gets trailing PAD.
    cut = len(audio) - PAUSE_BLOCKS // 2 * BLOCK
    np.testing.assert_array_equal(pieces[0], np.concatenate([audio[:cut], silence(PAD)]))
    assert len(pieces[0]) <= LIMIT
    assert chunker.finish() == []


@pytest.mark.parametrize("voiced_blocks", [MIN_VOICED - 1, MIN_VOICED])
def test_minimum_speech_duration(voiced_blocks):
    audio = np.concatenate([silence(BLOCK), noise(voiced_blocks * BLOCK)])
    pieces = collect(audio, 137)
    assert len(pieces) == int(voiced_blocks >= MIN_VOICED)
    if pieces:
        np.testing.assert_array_equal(pieces[0], np.concatenate([audio, silence(PAD)]))


def test_hard_cut_caps_padding_to_limit():
    # Make the final block the quietest of the last three seconds, but still voiced.
    audio = np.concatenate([
        silence(BLOCK), noise((MAX_BLOCKS - 2) * BLOCK), noise(BLOCK, level=0.02),
    ])
    chunker = Chunker()
    pieces = chunker.push(audio)
    assert len(pieces) == 1
    assert len(pieces[0]) == LIMIT
    assert 0 < LIMIT - len(audio) < PAD
    np.testing.assert_array_equal(
        pieces[0], np.concatenate([audio, silence(LIMIT - len(audio))]),
    )
    assert chunker.finish() == []


def test_long_continuous_speech_is_split_and_bounded():
    audio = np.concatenate([silence(10 * BLOCK), noise(25 * SR)])
    pieces = collect(audio, 4096)
    assert len(pieces) >= 3
    for piece in pieces:
        assert 0 < len(piece) <= LIMIT  # Includes all trailing padding.
        assert piece.dtype == np.float32
        assert np.any(piece != 0)
        assert piece[-1] == 0


@pytest.mark.parametrize("chunk_size", [137, BLOCK, 4096, SR + 7])
def test_push_chunk_size_does_not_change_pieces(chunk_size):
    audio = np.concatenate([
        noise(10 * BLOCK, level=0.0005),
        noise(8 * SR),
        silence(SR),
        noise(25 * SR),
        silence(SR),
        noise(SR + 123),
    ])
    expected = collect(audio, len(audio))
    actual = collect(audio, chunk_size)
    assert len(actual) == len(expected) >= 4
    for piece, reference in zip(actual, expected):
        np.testing.assert_array_equal(piece, reference)
        assert len(piece) <= LIMIT


def test_finish_preserves_pending_partial_block_and_drains_buffer():
    audio = np.concatenate([silence(BLOCK), noise(MIN_VOICED * BLOCK + 137)])
    chunker = Chunker()
    assert chunker.push(audio) == []
    pieces = chunker.finish()
    assert len(pieces) == 1
    np.testing.assert_array_equal(pieces[0], np.concatenate([audio, silence(PAD)]))
    assert len(pieces[0]) <= LIMIT
    assert chunker.finish() == []


def test_rms_avoids_integer_overflow():
    assert rms(np.array([30000, -30000], dtype=np.int16)) == 30000.0
