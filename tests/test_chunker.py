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
    """Steady noise: speech-like for short stretches, background when it lasts."""
    return np.random.default_rng(42).normal(0, level, samples).astype(np.float32)


def speech(samples, level=0.04):
    """Noise with a quiet 30 ms gap every 240 ms, like the gaps between syllables
    (without them a long stretch is steady background to the Chunker)."""
    x = noise(samples, level)
    for s in range(7 * BLOCK, samples, 8 * BLOCK):
        x[s:s + BLOCK] *= 0.05
    return x


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
        silence(10 * BLOCK), speech(196 * BLOCK), silence(PAUSE_BLOCKS * BLOCK),  # ends voiced
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
    # Make the final block the quietest of the last three seconds.
    audio = np.concatenate([
        silence(BLOCK), speech((MAX_BLOCKS - 2) * BLOCK), noise(BLOCK, level=0.001),
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
    audio = np.concatenate([silence(10 * BLOCK), speech(25 * SR)])
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
        speech(8 * SR),
        silence(SR),
        speech(25 * SR),
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


def test_speech_from_the_first_block_is_heard():
    # Steady speech-level audio with no quiet lead-in (the noise floor must not
    # start at the speech level).
    pieces = collect(noise(SR, level=0.04), 480)
    assert len(pieces) == 1
    assert len(pieces[0]) == SR + PAD


def test_short_last_piece_after_a_forced_cut_is_kept():
    # A forced cut at 10 s, then 5 voiced blocks (150 ms, below MIN_VOICED): the
    # tail continues speech that was already heard, so it must not be dropped.
    # The quietest block of the last 3 s is the last one before the cut, so the
    # cut lands exactly there (as in test_hard_cut_caps_padding_to_limit).
    audio = np.concatenate([silence(BLOCK), speech((MAX_BLOCKS - 2) * BLOCK), noise(BLOCK, level=0.001),
                            noise(5 * BLOCK)])
    pieces = collect(audio, 4096)
    assert len(pieces) == 2
    np.testing.assert_array_equal(pieces[1][-PAD:], silence(PAD))
    tail = pieces[1][:-PAD]
    np.testing.assert_array_equal(tail, audio[len(audio) - len(tail):])  # the end of the recording
    assert len(tail) == 5 * BLOCK


def test_silence_after_speech_is_still_dropped():
    audio = np.concatenate([silence(BLOCK), noise(SR), silence(12 * SR)])
    pieces = collect(audio, 4096)
    assert len(pieces) == 1


def test_quiet_speech_from_the_first_block_is_heard():
    pieces = collect(noise(SR, level=0.008), 480)
    assert len(pieces) == 1


def test_steady_background_becomes_the_noise_floor():
    # A constant hum at speech level: at most the first seconds (before the floor
    # adapts) may form a piece; the rest must not.
    pieces = collect(noise(12 * SR, level=0.02), 4096)
    assert len(pieces) <= 1
    assert sum(len(p) for p in pieces) <= 7 * SR
